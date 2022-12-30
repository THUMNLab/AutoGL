from .rl import *
import numpy as np
class AGNNReinforceController(ReinforceController):
    def resample(self,search_fields,selection):
        # search_fields act as group of fields in the paper (like activation group)
        self._initialize()
        result = selection.copy()

        # 1. update initial state with fields not searched
        for field in self.fields:
            if field not in search_fields:
                self._update_state(field,selection[field.name])

        # 2. get probability of field to search
        for field in search_fields:
            result[field.name] = self._sample_single(field)
        return result

    def _update_state(self,field,sampled):
        # use unsearched fields as rnn history to update state
        self._lstm_next_step()
        self._inputs = self.embedding[field.name](torch.LongTensor([sampled]).to(self._inputs.device))

class AGNNActionGuider(nn.Module):
    def __init__(self, fields, groups,guide_type, **controllargs):
        super(AGNNActionGuider, self).__init__()
        # create independent controllers for each group 
        controllers=[AGNNReinforceController(fields,**controllargs) for group in groups]
        self.controllers=nn.ModuleList(controllers)
        self.fields=fields
        self.groups=groups
        self.guide_type = guide_type

    def dummy_selection(self):
        # create dummy selection 
        result=dict()
        for field in self.fields:
            result[field.name]=0
        return result

    def resample(self,selection):
        entropys=[]
        new_selections=[]
        sample_probs=[]
        for idx,cont in enumerate(self.controllers):
            cont=self.controllers[idx]
            group=self.groups[idx]
            new_selection=cont.resample(group,selection)
            new_selections.append(new_selection)
            entropy=cont.sample_entropy
            entropys.append(entropy)
            sample_probs.append(cont.sample_log_prob)
        print(f'$$entropys {entropys}')
        if self.guide_type==0:
            # use the most uncertain one 
            idx=np.argmax(entropys)
        elif self.guide_type==1:
            # or sample by using entropy 
            idx=torch.multinomial(F.softmax(torch.tensor(entropys),dim=0),1).item()
        else:
            assert False,f"Not implemented guide type {self.guide_type}"
        group=self.groups[idx]
        print(f'$$select group {group}')
        new_selection=new_selections[idx]
        self.sample_log_prob=sample_probs[idx]
        self.sample_entropy=entropys[idx]
        print(f'$$new selection {new_selection}')
        return new_selection

@register_nas_algo("agnn")
class AGNNRL(GraphNasRL):
    def __init__(self,guide_type=1,*args,**kwargs):
        super(AGNNRL, self).__init__(*args,**kwargs)
        self.guide_type = guide_type
    def search(self, space: BaseSpace, dset, estimator):
        self.model = space
        self.dataset = dset  # .to(self.device)
        self.estimator = estimator
        # replace choice
        self.nas_modules = []

        k2o = get_module_order(self.model)
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)

        # to device
        self.model = self.model.to(self.device)
        # fields
        self.nas_fields = [
            ReinforceField(
                name,
                len(module),
                isinstance(module, PathSamplingLayerChoice) or module.n_chosen == 1,
            )
            for name, module in self.nas_modules
        ]

        # create groups
        tags='op in act concat'.split()
        groups={tag:[] for tag in tags}
        for field in self.nas_fields:
            for tag in tags:
                if tag in field.name:
                    groups[tag].append(field)
        groups=[x for x in groups.values() if x]

        # controller
        self.controller = AGNNActionGuider(
            self.nas_fields,
            groups,
            self.guide_type,
            lstm_size=100,
            temperature=5.0,
            tanh_constant=2.5,
            **(self.ctrl_kwargs or {}),
        )
        self.ctrl_optim = torch.optim.Adam(
            self.controller.parameters(), lr=self.ctrl_lr
        )

        # init selection (acc,selection)
        self.best_selection=[0,self.controller.dummy_selection()]

        # train
        with tqdm(range(self.num_epochs), disable=self.disable_progress) as bar:
            for i in bar:
                l2 = self._train_controller(i)
                bar.set_postfix(reward_controller=l2)

        selection=self.export()

        # selections = [x[1] for x in self.hist]
        # candidiate_accs = [-x[0] for x in self.hist]
        # # print('candidiate accuracies',candidiate_accs)
        # selection = self._choose_best(selections)
        arch = space.parse_model(selection, self.device)
        print(selection,arch)
        return arch

    def _train_controller(self, epoch):
        self.model.eval()
        self.controller.train()
        self.ctrl_optim.zero_grad()
        rewards = []
        selections=[]
        # baseline = None
        baseline=self.best_selection[0]
        # diff: graph nas train 100 and derive 100 for every epoch(10 epochs), we just train 100(20 epochs). totol num of samples are same (2000)
        with tqdm(
            range(self.ctrl_steps_aggregate), disable=self.disable_progress
        ) as bar:
            for ctrl_step in bar:
                self._resample()
                selections.append(self.selection.copy())
                metric, loss, hardware_metric = self._infer(mask="val")
                reward = metric

                # bar.set_postfix(acc=metric,loss=loss.item())
                LOGGER.debug(f"{self.arch}\n{self.selection}\n{metric},{loss}")
                # diff: not do reward shaping as in graphnas code
                if (
                    self.hardware_metric_limit is None
                    or hardware_metric[0] < self.hardware_metric_limit
                ):
                    self.hist.append([-metric, self.selection])
                    self.allhist.append([-metric, self.selection])
                    if len(self.hist) > self.topk:
                        self.hist.sort(key=lambda x: x[0])
                        self.hist.pop()
                rewards.append(reward)

                if self.entropy_weight:
                    reward += (
                        self.entropy_weight * self.controller.sample_entropy.item()
                    )

                if not baseline:
                    baseline = reward
                else:
                    baseline = baseline * self.baseline_decay + reward * (
                        1 - self.baseline_decay
                    )

                loss = self.controller.sample_log_prob * (reward - baseline)
                self.ctrl_optim.zero_grad()
                loss.backward()

                self.ctrl_optim.step()

                bar.set_postfix(acc=metric, max_acc=max(rewards))

            # conserative explorer: update the best selection
            idx=np.argmax(rewards)
            best_reward=rewards[idx]
            best_selection=selections[idx]
            if best_reward>self.best_selection[0]:
                self.best_selection=[best_reward,best_selection]

            print(f'$$best selection: {self.best_selection}')
        LOGGER.info("epoch:{}, mean rewards:{}".format(epoch, sum(rewards) / len(rewards)))
        return sum(rewards) / len(rewards)

    def _resample(self):
        result = self.controller.resample(self.best_selection[1])
        self.arch = self.model.parse_model(result, device=self.device)
        self.selection = result

    def export(self):
        # self.controller.eval()
        # with torch.no_grad():
        #     return self.controller.resample()
        return self.best_selection[1]
