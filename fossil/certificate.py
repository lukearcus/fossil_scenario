"""
This module defines the Certificate class and its subclasses, which are used to guide
the learner and verifier components in the fossil library. Certificates are used to 
certify properties of a system, such as stability or safety. The module also defines 
functions for logging loss and accuracy during the learning process, and for checking 
that the domains and data are as expected for a given certificate.
"""
# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Generator, Type, Any

import torch
import copy
import numpy as np
from torch.optim import Optimizer

import fossil.control as control
import fossil.logger as logger
import fossil.learner as learner
from fossil.consts import ScenAppConfig, CertificateType, DomainNames, ScenAppStateKeys
import fossil.domains as domains


XD = DomainNames.XD.value
XD1 = DomainNames.XD1.value
XD2 = DomainNames.XD2.value
XI = DomainNames.XI.value
XU = DomainNames.XU.value
XS = DomainNames.XS.value
XG = DomainNames.XG.value
XG1 = DomainNames.XG1.value
XG2 = DomainNames.XG2.value
XG_BORDER = DomainNames.XG_BORDER.value
XG1_BORDER = DomainNames.XG1_BORDER.value
XG2_BORDER = DomainNames.XG2_BORDER.value
XS_BORDER = DomainNames.XS_BORDER.value
XF = DomainNames.XF.value
XNF = DomainNames.XNF.value
XR = DomainNames.XR.value  # This is an override data set for ROA in StableSafe
HAS_BORDER = (XG, XS)
BORDERS = (XG_BORDER, XS_BORDER)
ORDER = (XD, XI, XU, XS, XG, XG_BORDER, XS_BORDER, XF, XNF)

cert_log = logger.Logger.setup_logger(__name__)


def log_loss_acc(t, loss, accuracy, verbose):
    # cert_log.debug(t, "- loss:", loss.item())
    # for k, v in accuracy.items():
    #     cert_log.debug(" - {}: {}%".format(k, v))
    loss_value = loss.item() if hasattr(loss, "item") else loss
    cert_log.debug("{} - loss: {:.5f}".format(t, loss_value))

    for k, v in accuracy.items():
        cert_log.debug(" - {}: {:.3f}%".format(k, v))


def _set_assertion(required, actual, name):
    if required != actual:
        raise ValueError(
            "Required {} {} do not match actual domains {}. Missing: {}, Not required: {}".format(
                name, required, actual, required - actual, actual - required
            )
        )


class Certificate:
    """
    Base class for certificates, used to define new Certificates.
    Certificates are used to guide the learner and verifier components.
    Methods learn and get_constraints must be implemented by subclasses.

    Attributes:
        domains: (symbolic) domains of the system. This is a dictionary of domain names and symbolic domains as SMT
            formulae.
            These may be stored as separate attributes for each domain, or
            as a dictionary of domain names and domains. They should be accessed accordingly.
        bias: Should the network have bias terms for this certificate? (default: True)
    """

    bias = True

    def __init__(self, domains: dict[str:Any]) -> None:
        pass

    def learn(
        self,
        learner: learner.LearnerNN,
        optimizer: Optimizer,
        S: list,
        Sdot: list,
        f_torch=None,
    ) -> dict:
        """
        Learns a certificate.

        Args:
            learner: fossil learner object (inherits from torch.nn.Module )
            optimizer: torch optimiser object
            S: dict of tensors of data (keys are domain names the data corresponds to, e.g. XD, XI)
            Sdot: dict of tensors containing f(data) (keys are domain names the data corresponds to, e.g. XD, XI)
            f_torch: torch function that computes f(data) (optional, for control synthesis)

        Returns:
            dict: empty dictionary



        This function is called by the learner object. It uses the sample Pytorch data points S and Sdot to
        calculate a loss function that should be minimised so the certificate properties are satisfied. The
        learn function does not return anything, but updates the optimiser object through the optimiser.step()
        function (which in turn updates the learners weights.)

        For control synthesis, the f_torch function is passed to the certificate, which is used to recompute the
        dynamics Sdot from the data S at each loop, since the control synthesis changes with each iteration.
        """
        raise NotImplemented("Not implemented in " + self.__class__.__name__)

    def get_constraints(self, verifier, C, Cdot) -> tuple:
        """
        Returns (negation of) contraints for the certificate.
        The constraints are returned as a tuple of dictionaries, where each dictionary contains the constraints
        that should be verified together. For simplicity, as single dictionary may be returned, but it may be useful
        to verify the most difficult constraints last. If an earlier constraint is not satisfied, the later ones
        will not be checked.
        The dictionary keys are the domain names the constraints correspond to, e.g. XD, XI, XU.

        Logical operators are provided by the verifier object, e.g. _And, _Or, _Not, using the solver_fncts method,
        which returns a dictionary of functions. Eg. _And = verifier.solver_fncts()["And"].

        Example certificates assume that domains are in the form of SMT formulae, and that the certificate stores them
        as instance attributes from the __init__. User defined certificates may follow a different format, but should
        be consistent in how they are stored and accessed. They are passed to the certificate as a dictionary of
        domain names and symbolic domains as SMT formulae, this cannot be changed.

        Args:
            verifier: fossil verifier object
            C: SMT formula of Certificate
            Cdot: SMT formula of Certificate time derivative or one-step difference (for discrete systems)

        Returns:
            tuple: tuple of dictionaries of certificate conditons


        """
        raise NotImplemented("Not implemented in " + self.__class__.__name__)

    @staticmethod
    def _assert_state(domains, data):
        """Checks that the domains and data are as expected for this certificate.

        This function is an optional debugging tool, but is called within CEGIS so should not be removed or
        renamed, and should only raise an exception if the domains or data are not as expected.
        """
        pass


class Practical_Lyapunov(Certificate):
    """
    Certificies stability for CT and DT models
    bool LLO: last layer of ones in network
    XD: Symbolic formula of domain
    XG: Goal region (around origin)
    """

    bias = False

    def __init__(self, domains, config: ScenAppConfig) -> None:
        self.domain = domains[XD]
        self.llo = config.LLO
        self.control = config.CTRLAYER is not None
        self.D = config.DOMAINS
        self.beta = None
        self.T = config.SYSTEM.time_horizon
    
    def compute_state_loss(
            self, 
            V_I: torch.Tensor, 
            V_G: torch.Tensor,
            V_D: torch.Tensor,
            V_SD: torch.Tensor,
            V_D_lie: torch.Tensor,
            beta: torch.Tensor,
            Vdot: torch.Tensor, 
            indices: list,
            supp_samples: set,
            convex: bool
    ) -> tuple[torch.Tensor, dict]:
        """_summary_

        Args:
            V (torch.Tensor): Lyapunov samples over domain
            Vdot (torch.Tensor): Lyapunov derivative samples over domain
            circle (torch.Tensor): Circle

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        
        relu = torch.nn.ReLU()
        
        init_loss = V_I
        border_loss = -V_SD
        goal_loss = V_G-(V_I.min()+V_D.min())/2#minus since V_I<0
        state_loss = -V_D+beta
        
        margin = 1e-5
        
        init_con = relu(init_loss+margin).mean()
        border_con = relu(border_loss+margin).mean()
        state_con = relu(state_loss+margin).mean()
        goal_con = relu(goal_loss+margin).mean()

        psi_s = state_con+border_con+init_con+goal_con
        return psi_s

    def compute_loss(
            self, 
            V_I: torch.Tensor, 
            V_G: torch.Tensor,
            V_D: torch.Tensor,
            V_SD: torch.Tensor,
            V_D_lie: torch.Tensor,
            beta: torch.Tensor,
            Vdot: torch.Tensor, 
            indices: list,
            supp_samples: set,
            convex: bool
    ) -> tuple[torch.Tensor, dict]:
        """_summary_

        Args:
            V (torch.Tensor): Lyapunov samples over domain
            Vdot (torch.Tensor): Lyapunov derivative samples over domain
            circle (torch.Tensor): Circle

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """

        
        relu = torch.nn.ReLU()
        
        init_loss = V_I
        border_loss = -V_SD
        goal_loss = V_G-V_I.min()
        #goal_loss = V_G-(V_I.min()+V_D.min())/2#minus since V_I<0 #this gives better convergence and epsilon for nonabsorbing, but prefer to have it only in the warm start since no theoretical basis for it...
        state_loss = -V_D+beta
        
        margin = 1e-5
        
        init_con = relu(init_loss+margin).mean()
        
        border_con = relu(border_loss+margin).mean()
        state_con = relu(state_loss+margin).mean()
        goal_con = relu(goal_loss+margin).mean()
        psi_s = state_con+border_con+init_con+goal_con
        if True:
            req_diff = ((V_I.max()-beta)/self.T)

            # Code below for trying to get samples before V<beta
            Vdot_selected = []
            selected_inds = []
            curr_ind = 0
            for inds in indices["lie"]:
                try:
                    final_ind = inds[0]+torch.where(V_D_lie[inds]<beta)[0][0] 
                except IndexError:
                    final_ind = inds[-1]+1
                selected = range(inds[0],final_ind)
                selected_inds.append(range(curr_ind,curr_ind+len(selected))) 
                curr_ind += len(selected)
                Vdot_selected.append(Vdot[selected])
            Vdot_selected = torch.hstack(Vdot_selected)
            lie_loss = Vdot_selected+relu(req_diff)+margin
            if psi_s != 0:
                lie_loss = relu(lie_loss)
            
            valid_Vdot = True
            if len(lie_loss) == 0:
                loss = 0
                valid_Vdot = False
            
            subgrad = not convex

            if subgrad:
                if valid_Vdot:
                    supp_max = torch.tensor([-1.])
                    lie_max = lie_loss.max()
                    ind_lie_max = lie_loss.argmax()
                    loss = lie_max
                    sub_sample = -1
                    for i, elem in enumerate(selected_inds):
                        if ind_lie_max in elem:
                            sub_sample = i
                            break
                    for ind in supp_samples:
                        inds = selected_inds[ind]
                        if len(inds) > 0:
                            supp_max = torch.max(supp_max, lie_loss[inds].max())
                    supp_loss = supp_max
                    new_sub_samples = set([sub_sample])
                else:
                    supp_loss = 0
                    new_sub_samples = set()
            else:
                raise NotImplementedError
            loss = loss + psi_s
            if supp_loss != -1:
                supp_loss = supp_loss + psi_s
            goal_accuracy = (V_G<V_I.min()).count_nonzero().item()/len(V_G)
            dom_accuracy = (V_D>beta).count_nonzero().item()/len(V_D)
            if len(Vdot_selected) > 0:
                lie_accuracy = (Vdot_selected <= -req_diff).count_nonzero().item()/len(Vdot_selected)
            else:
                lie_accuracy = 0
            accuracy = {"goal_acc": goal_accuracy * 100, "domain_acc" : dom_accuracy*100, "lie_acc" :lie_accuracy*100}
        else:
            supp_loss = psi_s 
            loss = psi_s
            new_sub_samples = set()
            goal_accuracy = (V_G<V_I.min()).count_nonzero().item()/len(V_G)
            dom_accuracy = (V_D>beta).count_nonzero().item()/len(V_D)
            accuracy = {"goal_acc": goal_accuracy * 100, "domain_acc" : dom_accuracy*100}

        return loss, supp_loss, accuracy, new_sub_samples

    def learn(
        self,
        learner: learner.LearnerNN,
        optimizer: Optimizer,
        S: list,
        Sdot: list,
        Sind: list,
        times: list,
        best_loss: float,
        best_net: learner.LearnerNN,
        f_torch=None,
        convex=False
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: dict of tensors of data
        :param Sdot: dict of tensors containing f(data)
        :return: --
        """

        batch_size = len(S[XD])
        learn_loops = 1000
        samples = S[XD]
        
        i1 = S[XD].shape[0]
        idot1 = Sdot[XD].shape[0]
        
        i2 = S[XI].shape[0]
        idot2 = Sdot[XI].shape[0]

        i3 = S[XG_BORDER].shape[0]
        idot3 = len(Sdot[XG_BORDER])

        i4 = S[XG].shape[0]
        idot4 = len(Sdot[XG])

        idot5 = len(Sdot[XS_BORDER])

        samples = torch.cat([S[XD], S[XI], S[XG_BORDER],  S[XG], S[XS_BORDER]])

        samples_dot = Sdot[XD]

        samples_with_nexts = samples[:idot1]
        states_only = torch.cat([samples[idot1:i1], samples[i1+idot2:i1+i2], samples[i1+i2+idot3:i1+i2+i3], samples[i1+i2+i3+idot4:i1+i2+i3+i4], samples[i1+i2+i3+i4+idot5:]])
        times = times[XD]

        supp_samples = set()
        state_sol = False

        for t in range(learn_loops):
            optimizer.zero_grad()
            if self.control:
                samples_dot = f_torch(samples)

            if state_sol:
                V1, Vdot, circle = learner.get_all(samples_with_nexts, samples_dot, times) 
                V2 = learner(states_only)
                V = V2
                V_D = V[:i1-idot1]
                V_I = V[i1-idot1:i1+i2-idot1-idot2]
                V_SG = V[i1+i2-idot1-idot2:i1+i2+i3-idot1-idot2-idot3]
                V_G = V[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
                V_SD = V[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]
                beta = V_SG.min()
                loss, supp_loss, learn_accuracy, sub_sample = self.compute_loss(V_I, V_G, V_D, V_SD, V1, beta, Vdot, Sind, supp_samples, convex)
                if loss <= best_loss:
                    best_loss = loss
                    best_net = copy.deepcopy(learner)
                    best_net.beta = beta.item()

                if self.control:
                    loss = loss + control.cosine_reg(samples, samples_dot)

                if t % 100 == 0 or t == learn_loops - 1:
                    log_loss_acc(t, loss, learn_accuracy, learner.verbose)
                if convex:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                    grads = torch.hstack([torch.flatten(param.grad) for param in learner.parameters()])
                    if supp_loss != -1:
                        optimizer.zero_grad()
                        supp_loss.backward(retain_graph=True)
                        supp_grads = torch.hstack([torch.flatten(param.grad) for param in learner.parameters()])
                        inner = torch.inner(grads, supp_grads)
                        if inner <= 0:
                            supp_samples = supp_samples.union(sub_sample)
                            optimizer.zero_grad()
                            loss.backward()
                    else:
                        supp_samples = supp_samples.union(sub_sample)
                optimizer.step()

                if learner._take_abs:
                    learner.make_final_layer_positive()
            else:
                state_itt = 0
                while True:
                    optimizer.zero_grad()
                    V1, Vdot, circle = learner.get_all(samples_with_nexts, samples_dot, times) 
                    V2 = learner(states_only)
                    V = V2
                    V_D = V[:i1-idot1]
                    V_I = V[i1-idot1:i1+i2-idot1-idot2]
                    V_SG = V[i1+i2-idot1-idot2:i1+i2+i3-idot1-idot2-idot3]
                    V_G = V[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
                    V_SD = V[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]
                    beta = V_SG.min()
                    state_itt += 1
                    loss = self.compute_state_loss(V_I, V_G, V_D, V_SD, V1, beta, Vdot, Sind, supp_samples, convex)
                    if loss == 0:
                        state_sol=True
                        break
                    else:
                        if state_itt % 1000 == 0:
                            loss_v = loss.item() if hasattr(loss, "item") else loss
                            cert_log.debug("{} - loss: {:.10f}".format(state_itt, loss_v))
                            
                        loss.backward()
                        optimizer.step()
                #loss = self.compute_state_loss(V_I, V_G, V_D, V_SD, V1, beta, Vdot, Sind, supp_samples, convex)
                #if loss == 0:
                #    state_sol = True
                #else:
                #    loss.backward()
                #    optimizer.step()
                
        V1, Vdot, circle = learner.get_all(samples_with_nexts, samples_dot, times)
        V2 = learner(states_only)
        V = V2
        V_D = V[:i1-idot1]
        V_I = V[i1-idot1:i1+i2-idot1-idot2]
        V_SG = V[i1+i2-idot1-idot2:i1+i2+i3-idot1-idot2-idot3]
        V_G = V[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
        V_SD = V[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]
        beta = V_SG.min()

        loss, supp_loss, learn_accuracy, sub_sample = self.compute_loss(V_I, V_G, V_D, V_SD, V1, beta, Vdot, Sind, supp_samples, convex)

        if self.control:
            loss = loss + control.cosine_reg(samples, samples_dot)
        if loss <= best_loss:
            best_loss = loss
            best_net = copy.deepcopy(learner)
            best_net.beta = beta.item()
        return {ScenAppStateKeys.loss: loss, "best_loss":best_loss, "best_net":best_net, "new_supps": supp_samples}

    def get_violations(self, V, Vdot, S, Sdot, times, state_data):
        req_diff = (V(state_data["init"]).max()-V(state_data["goal_border"]).min())/self.T
        violated = 0
        true_violated = 0
        for i, (traj, traj_deriv, time) in enumerate(zip(S, Sdot, times)):
            traj, traj_deriv, time = torch.tensor(traj.T, dtype=torch.float32), torch.tensor(np.array(traj_deriv).T, dtype=torch.float32), torch.tensor(time, dtype=torch.float32)
            
            valid_inds = torch.where(self.D[XD].check_containment(traj))
            traj = traj[valid_inds]
            traj_deriv = traj_deriv[valid_inds]
            time = time[valid_inds]
    
            pred_V = V(traj)
            pred_0 = V(torch.zeros_like(traj))
            pred_Vdot = Vdot(traj, traj_deriv, time)
            non_goal_inds = torch.where(domains.Complement(self.D[XG]).check_containment(traj))
            if len(non_goal_inds) == 0:
                true_violated += 1
            # We should check for value violations, but currently don't
            if any(pred_Vdot[non_goal_inds] > -req_diff):
                violated += 1
        return violated, true_violated

class Sequential_Reach(Certificate):
    """
    Certificies stability for CT and DT models
    bool LLO: last layer of ones in network
    XD: Symbolic formula of domain
    XG: Goal region (around origin)
    """

    bias = False

    def __init__(self, doms, config: ScenAppConfig) -> None:
        raise NotImplementedError # This whole class is not yet correctly implemented
        #self.domain = domains.Union(doms[XD1], doms[XD2])
        self.llo = config.LLO
        self.control = config.CTRLAYER is not None
        self.D = config.DOMAINS
        self.beta = None
        self.T1 = config.SYSTEM.T1
        self.T2 = config.SYSTEM.T2
        self.T = self.T1+self.T2

    def compute_loss(
            self, 
            V_I: torch.Tensor, 
            V_G1: torch.Tensor,
            V_D1: torch.Tensor,
            V_D_lie_1: torch.Tensor,
            V_G2: torch.Tensor,
            V_D2: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor,
            Vdot1: torch.Tensor, 
            Vdot2: torch.Tensor, 
            indices: list,
            supp_samples: set,
            convex: bool
    ) -> tuple[torch.Tensor, dict]:
        """_summary_

        Args:
            V (torch.Tensor): Lyapunov samples over domain
            Vdot (torch.Tensor): Lyapunov derivative samples over domain
            circle (torch.Tensor): Circle

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        relu = torch.nn.ReLU()
        init_loss = -V_I
        goal1_loss = V_G1
        goal2_loss = V_G2
        state1_loss = -V_D1+alpha
        state2_loss = -V_D2+beta
        margin = 1e-5
        req_diff1 = ((V_I.max()-alpha)/self.T1)
        req_diff2 = ((alpha-beta)/self.T2)
        lie2_loss = relu(Vdot2+relu(req_diff2))
        
        lie1_index = torch.nonzero(V_D_lie_1 > alpha)
         

        subgrad = not convex
        
        if subgrad:
            supp_max = torch.tensor([-1.])
            
            if lie1_index.nelement() != 0:
                Vdot_selected = torch.index_select(Vdot1, dim=0, index=lie1_index[:, 0])
                lie1_loss = relu(Vdot_selected+relu(req_diff1))
                lie1_max = lie1_loss.max()
                ind_lie1_max = lie1_loss.argmax()
                lie2_max = lie2_loss.max()
                ind_lie2_max = lie2_loss.argmax()
                loss = torch.max(lie1_max, lie2_max)
            else:
                lie1_max=1
                lie2_max = lie2_loss.max()
                ind_lie2_max = lie2_loss.argmax()
                loss = lie2_max

            if loss == lie1_max:
                ind_lie_max = ind_lie1_max
                indexer = indices["lie1"]
            else:
                ind_lie_max = ind_lie2_max
                indexer = indices["lie2"]
            sub_sample = -1
            for i, elem in enumerate(indexer):
                if ind_lie_max in elem:
                    sub_sample = i
                    break
            for ind in supp_samples:
                if lie1_index.nelement() != 0:
                    inds1 = indices["lie1"][ind]
                    adjusted_inds1 = torch.cat([torch.where(lie1_index[:,0] == elem)[0] for elem in inds1])
                else:
                    adjusted_inds1 = []
                inds2 = indices["lie2"][ind]
                #adjusted_inds2 = torch.cat([torch.where(lie2_index[:,0] == elem)[0] for elem in inds2])
                if len(adjusted_inds1) > 0:
                    supp_max = torch.max(supp_max, lie1_loss[adjusted_inds1].max())
                if len(inds2) > 0:
                    supp_max = torch.max(supp_max, lie2_loss[inds2].max())
            supp_loss = supp_max
            new_sub_samples = set([sub_sample])
        else:
            raise NotImplementedError
        goal_accuracy = ((V_G1<0).count_nonzero().item()+(V_G2<0).count_nonzero().item())/(len(V_G1)+len(V_G2))
        dom_accuracy = ((V_D1>alpha).count_nonzero().item()+(V_D2>beta).count_nonzero().item())/(len(V_D1)+len(V_D2))
        lie_accuracy = ((Vdot1 <= -req_diff1).count_nonzero().item() + (Vdot2<=-req_diff2).count_nonzero().item()) /(len(Vdot1)+len(Vdot2))
        accuracy = {"goal_acc": goal_accuracy * 100, "domain_acc" : dom_accuracy*100, "lie_acc": lie_accuracy*100}
        gamma = 1
        # init and goal constraints shouldn't be needed but speed up convergence
        init_con = relu(init_loss+margin).mean()
        goal1_con = relu(goal1_loss+margin).mean() 
        state1_con = relu(state1_loss+margin).mean()
        goal2_con = relu(goal2_loss+margin).mean() 
        state2_con = relu(state2_loss+margin).mean()
        loss = loss+ gamma*(state1_con+init_con+goal1_con+state2_con+goal2_con)
        if supp_loss != -1:
            supp_loss = supp_loss + gamma*(state1_con+init_con+goal1_con+state2_con+goal2_con)
        return loss, supp_loss, accuracy, new_sub_samples

    def learn(
        self,
        learner: learner.LearnerNN,
        optimizer: Optimizer,
        S: list,
        Sdot: list,
        Sind: list,
        times: list,
        best_loss: float,
        best_net: learner.LearnerNN,
        f_torch=None,
        convex=False
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: dict of tensors of data
        :param Sdot: dict of tensors containing f(data)
        :return: --
        """

        learn_loops = 1000
        
        D1_dot_index = 0, Sdot[XD1].shape[0]
        D1_index = Sdot[XD1].shape[0], S[XD1].shape[0]

        D2_dot_index = D1_index[1], D1_index[1]+Sdot[XD2].shape[0]
        D2_index = D2_dot_index[1], D2_dot_index[1]+S[XD2].shape[0]
        
        I_index = D2_index[1]+Sdot[XI].shape[0], D2_index[1]+S[XI].shape[0]
        
        G1Border_index = I_index[1]+len(Sdot[XG1_BORDER]), I_index[1]+S[XG1_BORDER].shape[0]
        
        G2Border_index = G1Border_index[1]+len(Sdot[XG2_BORDER]), G1Border_index[1]+S[XG2_BORDER].shape[0]
        
        G1_index = G1Border_index[1]+len(Sdot[XG1]), G1Border_index[1]+S[XG1].shape[0]
        
        G2_index = G1_index[1]+len(Sdot[XG2]), G1_index[1]+S[XG2].shape[0]

        samples = torch.cat([S[XD1], S[XD2], S[XI], S[XG1_BORDER], S[XG2_BORDER],  S[XG1], S[XG2]])

        samples_dot = torch.cat([Sdot[XD1], Sdot[XD2]])

        samples_with_nexts = torch.cat([samples[D1_dot_index[0]:D1_dot_index[1]], samples[D2_dot_index[0]:D2_dot_index[1]]])
        times = torch.cat([times[XD1], times[XD2]])

        supp_samples = set()
        for t in range(learn_loops):
            optimizer.zero_grad()
            if self.control:
                samples_dot = f_torch(samples)
            V1, Vdot, circle = learner.get_all(samples_with_nexts, samples_dot, times) # error here after discarding
            Vdot1 = Vdot[:D1_dot_index[1]]
            V_D_lie_1 = Vdot[:D1_dot_index[1]]
            Vdot2 = Vdot[D1_dot_index[1]:]
            V2 = learner(samples)
            V = V2
            V_D1 = V[D1_index[0]:D1_index[1]]
            V_D2 = V[D2_index[0]:D2_index[1]]
            V_I = V[I_index[0]:I_index[1]]
            V_SG1 = V[G1Border_index[0]:G1Border_index[1]]
            V_SG2 = V[G2Border_index[0]:G2Border_index[1]]
            V_G1 = V[G1_index[0]:G1_index[1]]
            V_G2 = V[G2_index[0]:G2_index[1]]
            alpha = V_SG1.min()
            beta = V_SG2.min()

            loss, supp_loss, learn_accuracy, sub_sample = self.compute_loss(V_I, V_G1, V_D1, V_D_lie_1, V_G2, V_D2, alpha, beta, Vdot1, Vdot2, Sind, supp_samples, convex)
            if loss <= best_loss:
                best_loss = loss
                best_net = copy.deepcopy(learner)
                best_net.beta = beta.item()


            if self.control:
                loss = loss + control.cosine_reg(samples, samples_dot)

            if t % 100 == 0 or t == learn_loops - 1:
                log_loss_acc(t, loss, learn_accuracy, learner.verbose)
            
            if convex:
                loss.backward()
            else:
                loss.backward(retain_graph=True)
                grads = torch.hstack([torch.flatten(param.grad) for param in learner.parameters()])
                # Code below is for non-convex
                if supp_loss != -1:
                    optimizer.zero_grad()
                    supp_loss.backward(retain_graph=True)
                    supp_grads = torch.hstack([torch.flatten(param.grad) for param in learner.parameters()])
                    inner = torch.inner(grads, supp_grads)
                    if inner <= 0:
                        supp_samples = supp_samples.union(sub_sample)
                        optimizer.zero_grad()
                        loss.backward()
                else:
                    supp_samples = supp_samples.union(sub_sample)
            optimizer.step()

            if learner._take_abs:
                learner.make_final_layer_positive()
        V1, Vdot, circle = learner.get_all(samples_with_nexts, samples_dot, times)
        V_D_lie_1 = Vdot[:D1_dot_index[1]]

        Vdot1 = Vdot[:D1_dot_index[1]]
        Vdot2 = Vdot[D1_dot_index[1]:]
        V2 = learner(samples)
        V = V2
        V_D1 = V[D1_index[0]:D1_index[1]]
        V_D2 = V[D2_index[0]:D2_index[1]]
        V_I = V[I_index[0]:I_index[1]]
        V_SG1 = V[G1Border_index[0]:G1Border_index[1]]
        V_SG2 = V[G2Border_index[0]:G2Border_index[1]]
        V_G1 = V[G1_index[0]:G1_index[1]]
        V_G2 = V[G2_index[0]:G2_index[1]]
        alpha = V_SG1.min()
        beta = V_SG2.min()


        loss, supp_loss, learn_accuracy, sub_sample = self.compute_loss(V_I, V_G1, V_D1, V_D_lie_1, V_G2, V_D2, alpha, beta, Vdot1, Vdot2, Sind, supp_samples, convex)
        if self.control:
            loss = loss + control.cosine_reg(samples, samples_dot)
        if loss <= best_loss:
            best_loss = loss
            best_net = copy.deepcopy(learner)
            best_net.beta = beta.item()
        return {ScenAppStateKeys.loss: loss, "best_loss":best_loss, "best_net":best_net, "new_supps": supp_samples}

    def get_violations(self, V, Vdot, S, Sdot, times, state_data):
        req_diff = (V(state_data["init"]).max()-V(state_data["goal_border"]).min())/self.T
        violated = 0
        true_violated = 0
        for i, (traj, traj_deriv, time) in enumerate(zip(S, Sdot, times)):
            traj, traj_deriv, time = torch.tensor(traj.T, dtype=torch.float32), torch.tensor(np.array(traj_deriv).T, dtype=torch.float32), torch.tensor(time, dtype=torch.float32)
            
            valid_inds = torch.where(self.D[XD].check_containment(traj))
            traj = traj[valid_inds]
            traj_deriv = traj_deriv[valid_inds]
            time = time[valid_inds]
    
            pred_V = V(traj)
            pred_0 = V(torch.zeros_like(traj))
            pred_Vdot = Vdot(traj, traj_deriv, time)
            non_goal_inds = torch.where(domains.Complement(self.D[XG]).check_containment(traj))
            if len(non_goal_inds) == 0:
                true_violated += 1
            # We should check for value violations, but currently don't
            if any(pred_Vdot[non_goal_inds] > -req_diff):
                violated += 1
        return violated, true_violated

class BarrierAlt(Certificate):
    """
    Certifies Safety of a model  using Lie derivative everywhere.

    Works for continuous and discrete models.

    Arguments:
    domains {dict}: dictionary of string: domains pairs for a initial set, unsafe set and domain


    """

    def __init__(self, domains, config: ScenAppConfig) -> None:
        self.domain = domains[XD]
        self.initial_s = domains[XI]
        self.unsafe_s = domains[XU]
        self.bias = True
        self.D = config.DOMAINS
        self.T=config.SYSTEM.time_horizon

    def compute_state_loss(
        self,
        B_i: torch.Tensor,
        B_u: torch.Tensor,
        B_d: torch.Tensor,
        Bdot_d: torch.Tensor,
        indices: list,
        supp_samples: set,
        convex: bool,
    ) -> tuple[torch.Tensor, dict]:
        relu = torch.nn.ReLU()
        
        unsafe_margin = 1e-5
        init_loss = (relu(B_i).mean())
        unsafe_loss = relu(-B_u+unsafe_margin).mean()
        psi_s = init_loss + unsafe_loss
        return psi_s 


    def compute_loss(
        self,
        B_i: torch.Tensor,
        B_u: torch.Tensor,
        B_d: torch.Tensor,
        Bdot_d: torch.Tensor,
        indices: list,
        supp_samples: set,
        convex: bool,
    ) -> tuple[torch.Tensor, dict]:
        """Computes loss function for Barrier certificate.

        Also computes accuracy of the current model.

        Args:
            B_i (torch.Tensor): Barrier values for initial set
            B_u (torch.Tensor): Barrier values for unsafe set
            B_d (torch.Tensor): Barrier values for domain
            Bdot_d (torch.Tensor): Barrier derivative values for domain

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        learn_accuracy = (B_i <= 0).count_nonzero().item() + (
            B_u > 0
        ).count_nonzero().item()
        percent_accuracy_init_unsafe = learn_accuracy * 100 / (len(B_u) + len(B_i))
        relu = torch.nn.ReLU()

        req_diff = (B_u.min() - B_i.max())/self.T
        lie_margin = 1e-5
        lie_loss = Bdot_d-req_diff
        lie_accuracy = (
            100 * ((Bdot_d < req_diff).count_nonzero()).item() / Bdot_d.shape[0]
        )
        subgrad = not convex
        
        unsafe_margin = 1e-5
        init_loss = (relu(B_i).mean())
        unsafe_loss = relu(-B_u+unsafe_margin).mean()
        psi_s = init_loss + unsafe_loss
        if True:
            if subgrad:
                supp_max = torch.tensor([-1.0])
                lie_max = lie_loss.max() 
                ind_lie_max = lie_loss.argmax()
                loss = lie_max
                if psi_s > 0:
                    loss = relu(loss)
                sub_sample = -1
                for i, elem in enumerate(indices["lie"]):
                    if ind_lie_max in elem:
                        sub_sample = i
                        break
                for ind in supp_samples:
                    lie_inds = indices["lie"][ind]
                    if len(lie_inds) > 0:
                        supp_max = torch.max(supp_max, lie_loss[lie_inds].max())
                supp_loss = supp_max
                new_sub_samples = set([sub_sample])
            else:    
                # relaxed constraints below
                supp_loss = -1
                loss = 0
                sub_samples = {"active":0,"relaxed":0}
                i = 0
                for inds_lie in indices["lie"]:
                    curr_max = torch.tensor(-1.0)
                    elems_lie = lie_loss[inds_lie]
                    if len(elems_lie) > 0 :
                        curr_max = torch.max(curr_max, elems_lie.max())
                    if curr_max > 0:
                        loss += curr_max
                        sub_samples["relaxed"] += 1
                    elif curr_max > -0.01: #Some wiggle room for what counts as active
                        sub_samples["active"] += 1
                    i += 1
                new_sub_samples = sub_samples
            loss = loss + psi_s
            if supp_loss != -1:
                supp_loss = supp_loss + psi_s
        else:
            loss = psi_s
            supp_loss = psi_s
            new_sub_samples = set()

        accuracy = {
            "acc init unsafe": percent_accuracy_init_unsafe,
            "acc lie": lie_accuracy,
        }
        return loss, supp_loss, accuracy, new_sub_samples
    
    def learn(
        self,
        learner: learner.LearnerNN,
        optimizer: Optimizer,
        S: list,
        Sdot: list,
        Sind: list,
        times: list,
        best_loss: float,
        best_net: learner.LearnerNN,
        f_torch=None,
        convex=False,
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: dict of tensors of data
        :param Sdot: dict of tensors containing f(data)
        :return: --
        """

        learn_loops = 1000
        condition_old = False
        i1 = S[XD].shape[0]
        idot1 = Sdot[XD].shape[0]
        i2 = S[XI].shape[0]
        idot2 = Sdot[XI].shape[0]
        if type(Sdot[XU]) is not list:
            idot3 = Sdot[XU].shape[0]
        else:
            idot3 = 0
        label_order = [XD, XI, XU]
        samples = torch.cat([S[label] for label in label_order if type(S[label]) is not list])
        samples_with_nexts = torch.cat([samples[:idot1], samples[i1:i1+idot2], samples[i1+i2:i1+i2+idot3]])
        states_only = torch.cat([samples[idot1:i1], samples[i1+idot2:i1+i2], samples[i1+i2+idot3:]])
        times = torch.cat([times[label] for label in label_order if type(times[label]) is not list])
        samples_dot = torch.cat([Sdot[label] for label in label_order if type(Sdot[label]) is not list])
        supp_samples = set()
        state_sol = False
        for t in range(learn_loops):
            optimizer.zero_grad()


            B, Bdot, _ = learner.get_all(samples_with_nexts, samples_dot, times)
            
            B2 = learner(states_only)
            (
                B_d,
                Bdot_d,
            ) = (
                    B2[:i1-idot1] ,
                Bdot[:idot1],
            )
            B_i = B2[i1-idot1:i1+i2-idot2-idot1]
            B_u = B2[i1+i2-idot1-idot2:]
            if state_sol:
                loss, supp_loss, accuracy, sub_sample = self.compute_loss(B_i, B_u, B_d, Bdot_d, Sind, supp_samples, convex)
                
                if loss <= best_loss:
                    best_loss = loss
                    best_net = copy.deepcopy(learner)

                if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                    log_loss_acc(t, loss, accuracy, learner.verbose)

                if convex:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                    grads = torch.hstack([torch.flatten(param.grad) for param in learner.parameters()])
                    # Code below is for non-convex
                    if supp_loss != -1:
                        optimizer.zero_grad()
                        supp_loss.backward(retain_graph=True)
                        supp_grads = torch.hstack([torch.flatten(param.grad) for param in learner.parameters()])
                        inner = torch.inner(grads, supp_grads)
                        if inner <= 0:
                            supp_samples = supp_samples.union(sub_sample)
                            optimizer.zero_grad()
                            loss.backward()
                    else:
                        supp_samples = supp_samples.union(sub_sample)
                optimizer.step()
            else:
                state_itt = 0
                while True:
                    optimizer.zero_grad()
                    B, Bdot, _ = learner.get_all(samples_with_nexts, samples_dot, times)
                    
                    B2 = learner(states_only)
                    (
                        B_d,
                        Bdot_d,
                    ) = (
                            B2[:i1-idot1] ,
                        Bdot[:idot1],
                    )
                    B_i = B2[i1-idot1:i1+i2-idot2-idot1]
                    B_u = B2[i1+i2-idot1-idot2:]
                    state_itt += 1
                    loss = self.compute_state_loss(B_i, B_u, B_d, Bdot_d, Sind, supp_samples, convex)
                    if loss == 0:
                        state_sol=True
                        break
                    else:
                        if state_itt % 100 == 0:
                            loss_v = loss.item() if hasattr(loss, "item") else loss
                            cert_log.debug("{} - loss: {:.5f}".format(state_itt, loss_v))
                            
                        loss.backward()
                        optimizer.step()


        B, Bdot, _ = learner.get_all(samples_with_nexts, samples_dot, times)
        B2 = learner(states_only)
        (
            B_d,
            Bdot_d,
        ) = (
                B2[:i1-idot1] ,
            Bdot[:idot1],
        )
        B_i = B2[i1-idot1:i1+i2-idot2-idot1]
        B_u = B2[i1+i2-idot1-idot2:]
        loss, supp_loss, accuracy, sub_sample = self.compute_loss(B_i, B_u, B_d, Bdot_d, Sind, supp_samples, convex)
        
        if loss <= best_loss:
            best_loss = loss
            best_net = copy.deepcopy(learner)
            print(supp_samples)
        return {ScenAppStateKeys.loss: loss, "best_loss":best_loss, "best_net":best_net, "new_supps": supp_samples}

    def get_violations(self, B, Bdot, S, Sdot, times, state_data):
        req_diff = (B(state_data["unsafe"]).min()-B(state_data["init"]).max())/self.T
        true_violated = 0
        violated = 0
        for i, (traj, traj_deriv, time) in enumerate(zip(S, Sdot, times)):
            traj, traj_deriv, time = torch.tensor(traj.T, dtype=torch.float32), torch.tensor(np.array(traj_deriv).T, dtype=torch.float32), torch.tensor(time, dtype=torch.float32)

            valid_inds = torch.where(self.D[XD].check_containment(traj))
            
            traj = traj[valid_inds]
            traj_deriv = traj_deriv[valid_inds]
            time = time[valid_inds]

            initial_inds = torch.where(self.D[XI].check_containment(traj))
            unsafe_inds = torch.where(self.D[XU].check_containment(traj))
            pred_B_i = B(traj[initial_inds])
            pred_B_u = B(traj[unsafe_inds])
            pred_B_dots = Bdot(traj, traj_deriv, time)
            if any(self.D[XU].check_containment(traj)):
                true_violated += 1
            #if (any(pred_B_i >= 0) or
            #        any(pred_B_u <= 0)):
            #    raise ValueError("Value violation!")
            if any(pred_B_dots > req_diff):
                violated += 1
                continue
        return violated, true_violated



class RWS(Certificate):
    """Certificate to satisfy a reach-while-stay property.

    Reach While stay must satisfy:
    forall x in XI, V <= 0,
    forall x in boundary of XS, V > 0,
    forall x in A \ XG, dV/dt < 0
    A = {x \in XS| V <=0 }

    """

    def __init__(self, domains, config: ScenAppConfig) -> None:
        """initialise the RWS certificate
        Domains should contain:
            XI: compact initial set
            XS: compact safe set
            dXS: safe border
            XG: compact goal set
            XD: whole domain

        Data sets for learn should contain:
            SI: points from XI
            SU: points from XD \ XS
            SD: points from XS \ XG (domain less unsafe and goal set)

        """
        self.domain = domains[XD]
        self.initial = domains[XI]
        self.safe = domains[XS]
        self.safe_border = domains[XS_BORDER]
        self.goal = domains[XG]
        self.bias = True
        self.BORDERS = (XS,)
        self.D = config.DOMAINS
        self.T = config.SYSTEM.time_horizon

    def compute_state_loss(self, V_i, V_u, V_d, V_d_states, V_g, Vdot_d, beta, indices, supp_samples, convex):
        margin = 1e-5
        margin_lie = 0.0
        acc_init = (V_i <= -margin).count_nonzero().item()*100/len(V_i)
        acc_unsafe = (V_u >= margin).count_nonzero().item()*100/len(V_u)
        acc_domain = (V_d_states > beta).count_nonzero().item()*100/len(V_d_states)
        relu = torch.nn.ReLU()

        subgrad = not convex

        Vdot_selected = []
        selected_inds = []
        curr_ind = 0
        
        init_loss = relu(V_i + margin).mean()
        unsafe_loss = relu(-V_u + margin).mean()
        state_loss = relu(-V_d_states + beta+margin).mean()
        goal_loss = relu(V_g-(V_i.min()+V_d_states.min())/2+margin).mean()
        
        psi_s = init_loss+unsafe_loss+state_loss+goal_loss
        accuracy = {
        "acc init": acc_init,
        "acc unsafe": acc_unsafe,
        "acc domain": acc_domain,
        }
        return psi_s, accuracy

    def compute_loss(self, V_i, V_u, V_d, V_d_states, V_g, Vdot_d, beta, indices, supp_samples, convex):
        # V_d must match Vdot_d
        margin = 1e-5
        margin_lie = 0.0
        acc_init = (V_i <= -margin).count_nonzero().item()*100/len(V_i)
        acc_unsafe = (V_u >= margin).count_nonzero().item()*100/len(V_u)
        acc_domain = (V_d_states > beta).count_nonzero().item()*100/len(V_d_states)
        relu = torch.nn.ReLU()

        subgrad = not convex

        Vdot_selected = []
        selected_inds = []
        Vdot_unselected = []
        unselected_inds = []
        curr_ind = 0
        curr_un_ind = 0

        init_loss = relu(V_i + margin).mean()
        unsafe_loss = relu(-V_u + margin).mean()
        state_loss = relu(-V_d_states + beta+margin).mean()
        goal_loss = relu(V_g-V_i.min()+margin).mean()#minus since V_I<0
        
        psi_s = init_loss+unsafe_loss+state_loss+goal_loss
        
        if True:
            for inds in indices["lie"]:
                try:
                    final_ind = inds[0]+torch.where(V_d[inds]<beta)[0][0] 
                except IndexError:
                    final_ind = inds[-1]+1
                selected = range(inds[0],final_ind)
                unselected = range(final_ind, inds[-1])
                selected_inds.append(range(curr_ind,curr_ind+len(selected))) 
                unselected_inds.append(range(curr_un_ind, curr_un_ind+len(unselected)))

                curr_ind += len(selected)
                curr_un_ind += len(unselected)

                Vdot_selected.append(Vdot_d[selected])
                
                Vdot_unselected.append(Vdot_d[unselected])

            Vdot_selected = torch.hstack(Vdot_selected)
            Vdot_unselected = torch.hstack(Vdot_unselected)

            req_diff = relu((V_i.max()-beta)/self.T)
            
            lie_loss = Vdot_selected+req_diff
            
            if psi_s > 0:
                lie_loss = relu(lie_loss)

            req_diff_2 = relu((V_u.min()-beta)/self.T)

            barr_lie_loss=Vdot_unselected-req_diff_2
            
            if psi_s > 0:
                barr_lie_loss = relu(barr_lie_loss)



            valid_Vdot = True
            if len(lie_loss) == 0:
                lie_accuracy = 0.0
                loss = 0.0
                # plus 0.1 so this doesn't accidentally lead to loss = 0
                supp_loss = -1
                valid_Vdot = False

            if valid_Vdot:

                # this might need changing in case there are points in the unsafe or goal set?
                # ensure no goal states in domain data (see rwa_2 for example)
                lie_accuracy=(((Vdot_selected <= -req_diff).count_nonzero()).item() * 100 / Vdot_selected.shape[0]
                )
                if subgrad:
                    supp_max = torch.tensor([-1.0])
                    lie_max = lie_loss.max()
                    barr_lie_max = barr_lie_loss.max()
                    if lie_max > barr_lie_max:
                        ind_lie_max = lie_loss.argmax()
                        loss = lie_max
                        sub_sample = -1
                        for i, elem in enumerate(selected_inds):
                            if ind_lie_max in elem:
                                sub_sample = i
                                break
                        for ind in supp_samples:
                            lie_inds = selected_inds[ind]
                            #adjusted_inds = torch.cat([torch.where(lie_index[:,0] == elem)[0] for elem in lie_inds])
                            if len(lie_inds) > 0:
                                supp_max = torch.max(supp_max, lie_loss[lie_inds].max())
                    else:
                        ind_lie_max = barr_lie_loss.argmax()
                        loss = barr_lie_max
                        sub_sample = -1
                        for i, elem in enumerate(indices["lie"]):
                            if ind_lie_max in elem:
                                sub_sample=i
                                break
                        for ind in supp_samples:
                            lie_inds = unselected_inds[ind]
                            if len(lie_inds) > 0:
                                supp_max = torch.max(supp_max, barr_lie_loss[lie_inds].max())
                    supp_loss = supp_max
                    new_sub_samples = set([sub_sample])
                else:
                    raise NotImplementedError
            else:
                supp_loss = 0 
                new_sub_samples = set()
            loss = loss+psi_s
            if supp_loss != -1:
                supp_loss = supp_loss + psi_s
        else:
            loss = psi_s
            supp_loss = psi_s
            new_sub_samples = set()

        accuracy = {
            "acc init": acc_init,
            "acc unsafe": acc_unsafe,
            "acc domain": acc_domain,
            "acc lie": lie_accuracy,
        }

        return loss, supp_loss, accuracy, new_sub_samples

    def learn(
        self,
        learner: learner.LearnerNN,
        optimizer: Optimizer,
        S: list,
        Sdot: list,
        Sind: list,
        times: list,
        best_loss: float,
        best_net: learner.LearnerNN,
        f_torch = None,
        convex=False,
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: dict of tensors of data
        :param Sdot: dict of tensors containing f(data)
        :return: --
        """
        
        assert len(S) == len(Sdot)

        learn_loops = 1000
        condition_old = False
        i1 = S[XD].shape[0]
        idot1 = Sdot[XD].shape[0]
        i2 = S[XI].shape[0]
        idot2 = Sdot[XI].shape[0]
        
        i3 = S[XG].shape[0]
        idot3 = 0
        if type(Sdot[XS_BORDER]) is not list:
            idot4 = Sdot[XS_BORDER].shape[0]
        else:
            idot4 = 0
        i4 = S[XS_BORDER].shape[0]

        idot5 = len(Sdot[XG_BORDER])

        label_order = [XD, XI, XG, XS_BORDER, XG_BORDER]
        samples = torch.cat([S[label] for label in label_order if type(S[label]) is not list])

        samples_dot = Sdot[XD]
        samples_with_nexts = S[XD][:idot1]
        states_only = torch.cat([samples[idot1:i1], samples[i1+idot2:i1+i2], samples[i1+i2+idot3:i1+i2+i3], samples[i1+i2+i3+idot4:i1+i2+i3+i4], samples[i1+i2+i3+i4+idot5:]])
        times = torch.cat([times[label] for label in label_order if type(times[label]) is not list])
        supp_samples = set()
        state_sol = False

        for t in range(learn_loops):
            optimizer.zero_grad()

            B_d, Bdot_d, _ = learner.get_all(samples_with_nexts, samples_dot, times[:idot1])

            B = learner(states_only)
            
            B_d_states = B[:i1-idot1]
            B_i = B[i1-idot1 : i1 + i2-idot1-idot2]
            B_g = B[i1 + i2-idot1-idot2 :i1+i2+i3-idot1-idot2-idot3]
            B_u = B[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
            B_sg = B[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]
            beta = B_sg.min()
            
            if state_sol:
            
                loss, supp_loss, accuracy, sub_sample = self.compute_loss(B_i, B_u, B_d, B_d_states, B_g, Bdot_d, beta, Sind, supp_samples, convex)
                


                if loss <= best_loss:
                    best_loss = loss
                    best_net = copy.deepcopy(learner)
                    best_net.beta = beta.item()
                

                if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                    log_loss_acc(t, loss, accuracy, learner.verbose)
                if convex:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                    grads = torch.hstack([torch.flatten(param.grad) for param in learner.parameters()])
                    if supp_loss != -1:
                        optimizer.zero_grad()
                        supp_loss.backward(retain_graph=True)
                        supp_grads = torch.hstack([torch.flatten(param.grad) for param in learner.parameters()])
                        inner = torch.inner(grads, supp_grads)
                        if inner <= 0:
                            supp_samples = supp_samples.union(sub_sample)
                            optimizer.zero_grad()
                            loss.backward()
                    else:
                        supp_samples = supp_samples.union(sub_sample)
                optimizer.step()
            else:
                state_itt = 0
                while True:
                    optimizer.zero_grad()
                    B_d, Bdot_d, _ = learner.get_all(samples_with_nexts, samples_dot, times[:idot1])

                    B = learner(states_only)
                    
                    B_d_states = B[:i1-idot1]
                    B_i = B[i1-idot1 : i1 + i2-idot1-idot2]
                    B_g = B[i1 + i2-idot1-idot2 :i1+i2+i3-idot1-idot2-idot3]
                    B_u = B[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
                    B_sg = B[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]
                    beta = B_sg.min()
                    state_itt += 1
                    state_loss, accuracy = self.compute_state_loss(B_i, B_u, B_d, B_d_states, B_g, Bdot_d, beta, Sind, supp_samples, convex)
                    if state_loss == 0:
                        state_sol = True
                    else:
                        state_loss.backward()
                        optimizer.step()
                        if state_itt % 100 == 0:
                            loss_v = loss.item() if hasattr(loss, "item") else loss
                            cert_log.debug("{} - loss: {:.5f}".format(state_itt, loss_v))
                            
        B_d, Bdot_d, _ = learner.get_all(samples_with_nexts, samples_dot, times[:idot1])
        B = learner(states_only)
        B_d_states = B[:i1-idot1]
        B_i = B[i1-idot1:i1+i2-idot2-idot1]
        B_g = B[i1 + i2-idot1-idot2 :i1+i2+i3-idot1-idot2-idot3]
        B_u = B[i1+i2+i3-idot1-idot2-idot3:i1+i2+i3+i4-idot1-idot2-idot3-idot4]
        B_sg = B[i1+i2+i3+i4-idot1-idot2-idot3-idot4:]

        beta = B_sg.min()
        loss, supp_loss, accuracy, sub_sample = self.compute_loss(B_i, B_u, B_d, B_d_states, B_g, Bdot_d, beta, Sind, supp_samples, convex)
        
        if loss <= best_loss:
            best_loss = loss
            best_net = copy.deepcopy(learner)
            best_net.beta = beta.item()
        return {ScenAppStateKeys.loss: loss, "best_loss":best_loss, "best_net":best_net, "new_supps": supp_samples}

    def get_violations(self, B, Bdot, S, Sdot, times, states):
        req_diff = (B(states["init"]).max()-B(states["goal"]).min())/self.T
        true_violated = 0
        violated = 0
        for i, (traj, traj_deriv, time) in enumerate(zip(S, Sdot, times)):
            traj, traj_deriv, time = torch.tensor(traj.T, dtype=torch.float32), torch.tensor(np.array(traj_deriv).T, dtype=torch.float32), torch.tensor(time, dtype=torch.float32)
            initial_inds = torch.where(self.D[XI].check_containment(traj))

            goal_inds = torch.where(self.D[XG].check_containment(traj))


            V_d = B(traj)

            pred_B_dots = Bdot(traj, traj_deriv, time)
            
            goal_inds = torch.where(self.D[XG].check_containment(traj))[0]
            if not all(self.D[XS].check_containment(traj)) or not any(self.D[XG].check_containment(traj)):
                true_violated += 1
            lie_inds = torch.nonzero(V_d <= 0)
            if any(self.D[XG].check_containment(traj)):
                lie_inds = [ind.item() for ind in lie_inds if ind not in goal_inds]
            if any(pred_B_dots[lie_inds] > req_diff):
                violated += 1
                continue
        return violated, true_violated


    @staticmethod
    def _assert_state(domains, data):
        domain_labels = set(domains.keys())
        data_labels = set(data.keys())
        _set_assertion(
            set([XD, XI, XS, XS_BORDER, XG]), domain_labels, "Symbolic Domains"
        )
        _set_assertion(set([XD, XI, XU]), data_labels, "Data Sets")


class RSWS(RWS):
    """Reach and Stay While Stay Certificate

    Firstly satisfies reach while stay conditions, given by:
        forall x in XI, V <= 0,
        forall x in boundary of XS, V > 0,
        forall x in A \ XG, dV/dt < 0
        A = {x \in XS| V <=0 }

    http://arxiv.org/abs/1812.02711
    In addition to the RWS properties, to satisfy RSWS:
    forall x in border XG: V > \beta
    forall x in XG \ int(B): dV/dt <= 0
    B = {x in XS | V <= \beta}
    Best to ask SMT solver if a beta exists such that the above holds -
    but currently we don't train for this condition.

    Crucially, this relies only on the border of the safe set,
    rather than the safe set itself.
    Since the border must be positive (and the safe invariant negative), this is inline
    with making the complement (the unsafe set) positive. Therefore, the implementation
    requires an unsafe set to be passed in, and assumes its border is the same of the border of the safe set.
    """

    def __init__(self, domains, config: ScenAppConfig) -> None:
        """initialise the RSWS certificate
        Domains should contain:
            XI: compact initial set
            XS: compact safe set
            dXS: safe border
            XG: compact goal set
            XD: whole domain

        Data sets for learn should contain:
            SI: points from XI
            SU: points from XD \ XS
            SD: points from XS \ XG (domain less unsafe and goal set)

        """
        raise NotImplementedError #This class currently not implemented
        self.domain = domains[XD]
        self.initial = domains[XI]
        self.safe = domains[XS]
        self.safe_border = domains[XS_BORDER]
        self.goal = domains[XG]
        self.goal_border = domains[XG_BORDER]
        self.BORDERS = (XS, XG)
        self.bias = True
        self.D = config.DOMAINS
        self.T = config.SYSTEM.time_horizon

    def compute_beta_loss(self, beta, V_g_border_min, V_g, Vdot_g, V_d, indices, supp_samples, convex):
        """Compute the loss for the beta condition
        :param beta: the guess value of beta based on the min of V of XG_border
        :param V_d: the value of V at points in the goal set
        :param Vdot_d: the value of the lie derivative of V at points in the goal set"""
        lie_index = torch.nonzero(V_g <= V_g_border_min)
        
        relu = torch.nn.ReLU()

        req_diff = relu(V_g_border_min-beta)/self.T

        if lie_index.nelement() != 0:
            subgrad = not convex
            beta_lie = relu(torch.index_select(Vdot_g, dim=0, index=lie_index[:, 0])-req_diff)
            accuracy = (beta_lie <= 0).count_nonzero().item() * 100 / beta_lie.shape[0]
            if subgrad:
                supp_max = torch.tensor([-1.0])
                lie_max = beta_lie.max()
                ind_lie_max = beta_lie.argmax()
                beta_loss = lie_max
                sub_sample = -1
                for i, elem in enumerate(indices["lie"]):
                    if lie_index[ind_lie_max, 0] in elem:
                        sub_sample = i
                        break
                for ind in supp_samples:
                    lie_inds = indices["lie"][ind]
                    adjusted_inds = torch.cat([torch.where(lie_index[:,0] == elem)[0] for elem in lie_inds])
                    if len(adjusted_inds) > 0:
                        supp_max = torch.max(supp_max, beta_lie[adjusted_inds].max())
                supp_beta_loss = supp_max
                new_sub_samples = set([sub_sample])
            else:
                raise NotImplementedError
        else:
            # Do we penalise V > beta in safe set, or  V < beta in goal set?
            beta_loss = torch.Tensor([0])
            supp_beta_loss = -1 
            new_sub_samples = set()

        return beta_loss, supp_beta_loss, new_sub_samples

    def learn(
        self,
        learner: learner.LearnerNN,
        optimizer: Optimizer,
        S: list,
        Sdot: list,
        Sind: list,
        times: list,
        best_loss: float,
        best_net: learner.LearnerNN,
        f_torch=None,
        convex = False,
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: dict of tensors of data
        :param Sdot: dict of tensors containing f(data)
        :return: --
        """
        assert len(S) == len(Sdot)

        learn_loops = 1000
        
        lie_indices = Sdot[XD].shape[0], S[XD].shape[0]
        lie_dot_indices = 0, Sdot[XD].shape[0]

        init_indices = lie_indices[1]+Sdot[XI].shape[0], lie_indices[1] + S[XI].shape[0]

        unsafe_indices = init_indices[1]+len(Sdot[XU]), init_indices[1] + S[XU].shape[0]

        goal_border_indices = (
            unsafe_indices[1] + len(Sdot[XG_BORDER]),
            unsafe_indices[1] + S[XG_BORDER].shape[0],
        )

        goal_indices = (
            goal_border_indices[1]+len(Sdot[XG]),
            goal_border_indices[1] + S[XG].shape[0],
        )

        goal_dot_indices = goal_border_indices[1], goal_indices[0]
        # Setting label order allows datasets to be passed in any order
        label_order = [XD, XI, XU, XG_BORDER, XG]
        lie_label_order = [XD, XG] # make sure no goal data in XD
        samples = torch.cat([S[label] for label in label_order if type(S[label]) is not list])

        samples_dot = torch.cat([Sdot[label] for label in lie_label_order])
        
        samples_with_nexts = torch.cat([samples[:lie_dot_indices[1]], samples[goal_dot_indices[0]:goal_dot_indices[1]]])
        
        times = torch.cat([times[label] for label in label_order if type(times[label]) is not list])
        supp_samples = set()
        for t in range(learn_loops):
            optimizer.zero_grad()

            V, Vdot, _ = learner.get_all(samples_with_nexts, samples_dot, torch.cat([times[:lie_dot_indices[1]],times[-len(Sdot[XG]):]]))

            (
                V_d,
                gradV_d,
            ) = (V[: lie_dot_indices[1]], Vdot[: lie_dot_indices[1]])

            Vstates = learner(samples)

            V_d_states = Vstates[lie_indices[0]:lie_indices[1]]
            V_i = Vstates[init_indices[0] : init_indices[1]]
            V_u = Vstates[unsafe_indices[0] : unsafe_indices[1]]
            S_dg = samples[goal_border_indices[0] : goal_border_indices[1]]
            V_g = Vstates[goal_indices[0] : goal_indices[1]]
            
            Vdot_g = Vdot[lie_dot_indices[1] :]
            samples_dot_d = samples_dot[: lie_indices[1]]

            beta = learner.compute_minimum(S_dg)[0]+V_g.min()/100
            loss, supp_loss, accuracy, sub_sample  = self.compute_loss(V_i, V_u, V_d, V_d_states, V_g, gradV_d, beta, Sind, supp_samples, convex)

            beta2 = learner.compute_minimum(S_dg)[0]
            #beta_loss, supp_beta_loss, beta_sub_sample = 0, -1, set()
            # converges without beta loss
            beta_loss, supp_beta_loss, beta_sub_sample = self.compute_beta_loss(beta, beta2, V_g, Vdot_g, V_d_states, Sind, supp_samples, convex)
            loss = loss + beta_loss
            #loss = torch.max(loss,beta_loss)
            if supp_loss != -1:
                if supp_beta_loss != -1:   
                    supp_loss = supp_loss + supp_beta_loss
                    sub_sample = sub_sample.union(beta_sub_sample)
            else:
                supp_loss = supp_beta_loss
                sub_sample = beta_sub_sample

            if loss <= best_loss:
                best_loss = loss
                best_net = copy.deepcopy(learner)
                best_net.beta = beta.item()

            if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                log_loss_acc(t, loss, accuracy, learner.verbose)

            if convex:
                loss.backward()
            else:
                loss.backward(retain_graph=True)
                grads = torch.hstack([torch.flatten(param.grad) for param in learner.parameters()])
                if supp_loss != -1:
                    optimizer.zero_grad()
                    supp_loss.backward(retain_graph=True)
                    supp_grads = torch.hstack([torch.flatten(param.grad) for param in learner.parameters()])
                    inner = torch.inner(grads, supp_grads)
                    if inner <= 0:
                        supp_samples = supp_samples.union(sub_sample)
                        optimizer.zero_grad()
                        loss.backward()
                else:
                    supp_samples = supp_samples.union(sub_sample)
            optimizer.step()
        V, Vdot, _ = learner.get_all(samples_with_nexts, samples_dot, torch.cat([times[:lie_dot_indices[1]],times[-len(Sdot[XG]):]]))
        (
            V_d,
            gradV_d,
        ) = (V[: lie_dot_indices[1]], Vdot[: lie_dot_indices[1]])


        Vstates = learner(samples)
        
        V_d_states = Vstates[lie_indices[0] : lie_indices[1]]
        V_i = Vstates[init_indices[0] : init_indices[1]]
        V_u = Vstates[unsafe_indices[0] : unsafe_indices[1]]
        S_dg = samples[goal_border_indices[0] : goal_border_indices[1]]
        V_g = Vstates[goal_indices[0] : goal_indices[1]]
        
        Vdot_g = Vdot[lie_dot_indices[1] :]
        samples_dot_d = samples_dot[: lie_indices[1]]
        
        beta = learner.compute_minimum(S_dg)[0]+V_g.min()/100
        
        loss, supp_loss, accuracy, sub_sample  = self.compute_loss(V_i, V_u, V_d, V_d_states, V_g, gradV_d, beta, Sind, supp_samples, convex)

        beta2 = learner.compute_minimum(S_dg)[0]
        beta_loss, supp_beta_loss, beta_sub_sample = self.compute_beta_loss(beta, beta2, V_g, Vdot_g, V_d_states, Sind, supp_samples, convex)
        #loss = torch.max(loss, beta_loss)
        loss = loss + beta_loss

        if loss <= best_loss:
            best_loss = loss
            best_net = copy.deepcopy(learner)
            best_net.beta = beta.item()

        return {ScenAppStateKeys.loss: loss, "best_loss":best_loss, "best_net":best_net, "new_supps":supp_samples}


    def beta_search(self, learner, verifier, C, Cdot, S):
        return learner.compute_minimum(S[XG_BORDER])[0]
    
    def get_violations(self, B, Bdot, S, Sdot, times, states):
        req_diff = (B(states["init"]).max()-B(states["goal"]).min())/self.T
        true_violated = 0
        violated = 0
        for i, (traj, traj_deriv, time) in enumerate(zip(S, Sdot, times)):
            traj, traj_deriv, time = torch.tensor(traj.T, dtype=torch.float32), torch.tensor(np.array(traj_deriv).T, dtype=torch.float32), torch.tensor(time, dtype=torch.float32)

            #valid_inds = torch.where(self.D[XD].check_containment(traj))
            #
            #traj = traj[valid_inds]
            #traj_deriv = traj_deriv[valid_inds]
            #time = time[valid_inds]

            initial_inds = torch.where(self.D[XI].check_containment(traj))
            
            # getting too many violations, need to investigate

            goal_inds = torch.where(self.D[XG].check_containment(traj))


            V_d = B(traj)

            pred_B_dots = Bdot(traj, traj_deriv, time)
            
            goal_inds = torch.where(self.D[XG].check_containment(traj))[0]
            first_goal_ind = goal_inds[0]
            if not all(self.D[XS].check_containment(traj)) or not all(self.D[XG].check_containment(traj[first_goal_ind:])):
                true_violated += 1
            lie_inds = torch.nonzero(V_d <= 0)
            if any(self.D[XG].check_containment(traj)):
                lie_inds = [ind.item() for ind in lie_inds if ind not in goal_inds]
            if any(pred_B_dots[lie_inds] > req_diff):
                violated += 1
                continue
        return violated, true_violated
            


class DoubleCertificate(Certificate):
    """In Devel class for synthesising any two certificates together"""

    def __init__(self, domains, config: ScenAppConfig):
        self.certificate1 = None
        self.certificate2 = None

    def compute_loss(self, C1, C2, Cdot1, Cdot2):
        loss1 = self.certificate1.compute_loss(C1, Cdot1)
        loss2 = self.certificate2.compute_loss(C2, Cdot2)
        return loss1[0] + loss2[0]

    def learn(
        self, learner: tuple, optimizer: Optimizer, S: dict, Sdot: dict, f_torch=None
    ):
        pass

    def get_constraints(self, verifier, C, Cdot) -> Generator:
        """
        :param verifier: verifier object
        :param C: tuple containing SMT formula of Lyapunov function and barrier function
        :param Cdot: tuple containing SMT formula of Lyapunov lie derivative and barrier lie derivative

        """
        C1, C2 = C
        Cdot1, Cdot2 = Cdot
        cert1_cs = self.certificate1.get_constraints(verifier, C1, Cdot1)
        cert2_cs = self.certificate2.get_constraints(verifier, C2, Cdot2)
        for cs in (*cert1_cs, *cert2_cs):
            yield cs


class AutoSets:
    """Class for automatically handing sets for certificates"""

    def __init__(self, XD, certificate: CertificateType) -> None:
        self.XD = XD
        self.certificate = certificate

    def auto(self) -> (dict, dict):
        if self.certificate == CertificateType.PRACTICALLYAPUNOV:
            return self.auto_practical_lyap()
        elif self.certificate == CertificateType.SEQUENTIALREACH:
            return self.auto_sequential_reach()
        elif self.certificate == CertificateType.BARRIERALT:
            self.auto_barrier_alt(self.sets)
        elif self.certificate == CertificateType.RWS:
            self.auto_rws(self.sets)
        elif self.certificate == CertificateType.RSWS:
            self.auto_rsws(self.sets)
        elif self.certificate == CertificateType.RAR:
            self.auto_rar(self.sets)

    def auto_lyap(self) -> None:
        domains = {XD: self.XD}
        data = {XD: self.XD._generate_data(1000)}
        return domains, data


def get_certificate(
    certificate: CertificateType, custom_cert=None
) -> Type[Certificate]:
    if certificate == CertificateType.PRACTICALLYAPUNOV:
        return Practical_Lyapunov
    elif certificate == CertificateType.SEQUENTIALREACH:
        return Sequential_Reach
    elif certificate == CertificateType.BARRIERALT:
        return BarrierAlt
    elif certificate in (CertificateType.RWS, CertificateType.RWA):
        return RWS
    elif certificate in (CertificateType.RSWS, CertificateType.RSWA):
        return RSWS
    elif certificate == CertificateType.RAR:
        return ReachAvoidRemain
    elif certificate == CertificateType.CUSTOM:
        if custom_cert is None:
            raise ValueError(
                "Custom certificate not provided (use ScenAppConfig CUSTOM_CERTIFICATE)))"
            )
        return custom_cert
    else:
        raise ValueError("Unknown certificate type {}".format(certificate))
