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
XI = DomainNames.XI.value
XU = DomainNames.XU.value
XS = DomainNames.XS.value
XG = DomainNames.XG.value
XG_BORDER = DomainNames.XG_BORDER.value
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


class Lyapunov(Certificate):
    """
    Certificies stability for CT and DT models
    bool LLO: last layer of ones in network
    XD: Symbolic formula of domain

    """

    bias = False

    def __init__(self, domains, config: ScenAppConfig) -> None:
        self.domain = domains[XD]
        self.llo = config.LLO
        self.control = config.CTRLAYER is not None
        self.D = config.DOMAINS
        self.beta = None
        self.T = config.SYSTEM.time_horizon

    def compute_loss(
            self, 
            V: torch.Tensor, 
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
        #margin = 0.1 # Need to change this and carry it around for later
        slope = 10 ** (learner.LearnerNN.order_of_magnitude(Vdot.detach().abs().max()))
        #relu = torch.nn.LeakyReLU(1 / slope.item())
        relu = torch.nn.ReLU()
        # relu = torch.nn.Softplus()
        # compute loss function. if last layer of ones (llo), can drop parts with V
        state_loss = -V
        margin = 1e-3
        lie_loss = relu(Vdot+margin)

        subgrad = not convex
        
        if subgrad:
            supp_max = torch.tensor([-1.])
            lie_max = lie_loss.max()
            ind_lie_max = lie_loss.argmax()
            loss = lie_max
            sub_sample = -1
            for i, elem in enumerate(indices["lie"]):
                if ind_lie_max in elem:
                    sub_sample = i
                    break
            for ind in supp_samples:
                inds = indices["lie"][ind]
                supp_max = torch.max(supp_max, lie_loss[inds].max())
            supp_loss = supp_max
            new_sub_samples = set([sub_sample])
        else:
            loss = 0
            for inds in indices["lie"]:
                curr_max = torch.tensor(0.)
                if self.llo:
                    state_elems = torch.tensor(0.)
                else:
                    state_elems = state_loss[inds]
                lie_elems = lie_loss[inds]
                loss += lie_elems.max()
            #if self.llo:
            #    learn_accuracy = (Vdot <= -margin).count_nonzero().item()
            #    #loss = (relu(Vdot + margin * circle)).max()
            #    loss = (relu(Vdot + margin)).max()
            #else:
            #    learn_accuracy = 0.5 * (
            #        (Vdot <= -margin).count_nonzero().item()
            #        + (V >= margin).count_nonzero().item()
            #    )
            #    #loss = torch.max((relu(Vdot + margin * circle)).max() , (
            #    #    relu(-V + margin * circle)
            #    #).max()) # Why times circle?
            #    loss = torch.max((relu(Vdot + margin )).max() , (
            #        relu(-V + margin)
            #    ).max()) # Why times circle?
        accuracy = (V > 0).count_nonzero().item() + (Vdot < 0).count_nonzero().item()
        accuracy /= (len(V) + len(Vdot))
        accuracy = {"acc": accuracy * 100}
        gamma = 0.1 
        state_con = relu(state_loss+margin).mean()
        loss = loss+ gamma*state_con
        if supp_loss != -1:
            supp_loss = supp_loss + gamma*state_con
        #try:
        #    final_ind = [ind for ind in indices["lie"] if len(ind) > 0][-1][-1]
        #except IndexError:
        #    final_ind = -1
        #if final_ind < len(lie_loss) - 1:
        #    loss += relu(lie_loss)[final_ind+1:].sum() + relu(state_loss)[final_ind+1:].sum
        #    if supp_loss != -1:
        #        supp_loss += relu(lie_loss)[final_ind+1:].sum() + relu(state_loss)[final_ind+1:].sum()


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

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            samples_dot = Sdot[XD]

        idot = len(samples_dot)
        samples_with_nexts = samples[:idot]
        states_only = samples[idot:]
        times = times[XD]

        supp_samples = set()
        for t in range(learn_loops):
            optimizer.zero_grad()
            if self.control:
                samples_dot = f_torch(samples)

            V1, Vdot, circle = learner.get_all(samples_with_nexts, samples_dot, times)
            V2 = learner(states_only)
            #V = torch.cat([V1,V2])
            V = V2
            V -= learner(torch.zeros_like(samples))

            loss, supp_loss, learn_accuracy, sub_sample = self.compute_loss(V, Vdot, Sind, supp_samples, convex)
            if loss <= best_loss:
                best_loss = loss
                best_net = copy.deepcopy(learner)

            if self.control:
                loss = loss + control.cosine_reg(samples, samples_dot)

            if t % 100 == 0 or t == learn_loops - 1:
                log_loss_acc(t, loss, learn_accuracy, learner.verbose)

            # t>=1 ensures we always have at least 1 optimisation step
            if learn_accuracy["acc"] == 100 and t >= 1:
                break
            
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
                    #print(inner)
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
        V2 = learner(states_only)
        #V = torch.cat([V1,V2])
        V = V2
        V -= learner(torch.zeros_like(samples))
        loss, supp_loss, learn_accuracy, sub_sample = self.compute_loss(V, Vdot, Sind, supp_samples, convex)

        if self.control:
            loss = loss + control.cosine_reg(samples, samples_dot)
        if loss <= best_loss:
            best_loss = loss
            best_net = copy.deepcopy(learner)
        return {ScenAppStateKeys.loss: loss, "best_loss":best_loss, "best_net":best_net, "new_supps": supp_samples}

    def get_violations(self, V, Vdot, S, Sdot, times):
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
            if np.linalg.norm(traj[:, -1]) > 0.01: # check this does what I want it to do...
                true_violated += 1
            if any(pred_V < pred_0):
                raise ValueError("Value violation!")
            if any(pred_Vdot > 0):
                violated += 1
        return violated, true_violated


    def get_constraints(self, verifier, V, Vdot) -> Generator:
        """
        :param verifier: verifier object
        :param V: SMT formula of Lyapunov Function
        :param Vdot: SMT formula of Lyapunov lie derivative
        :return: tuple of dictionaries of lyapunov conditons
        """
        _Or = verifier.solver_fncts()["Or"]
        _And = verifier.solver_fncts()["And"]
        _Not = verifier.solver_fncts()["Not"]

        if self.llo:
            # V is positive definite by construction
            lyap_negated = Vdot >= 0
        else:
            lyap_negated = _Or(V <= 0, Vdot >= 0)

        not_origin = _Not(_And(*[xi == 0 for xi in verifier.xs]))
        lyap_negated = _And(lyap_negated, not_origin)
        lyap_condition = _And(self.domain, lyap_negated)
        for cs in ({XD: lyap_condition},):
            yield cs


    def estimate_beta(self, net):
        # This function is unused I think
        print("Estimating beta!")
        try:
            border_D = self.D[XD].sample_border(300)
            beta, _ = net.compute_minimum(border_D)
        except NotImplementedError:
            beta = self.D[XD].generate_data(300)
        return beta

    @staticmethod
    def _assert_state(domains, data):
        domain_labels = set(domains.keys())
        data_labels = set(data.keys())
        _set_assertion(set([XD]), domain_labels, "Symbolic Domains")
        _set_assertion(set([XD]), data_labels, "Data Sets")

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

    def compute_loss(
            self, 
            V_I: torch.Tensor, 
            V_G: torch.Tensor,
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
        #margin = 0.1 # Need to change this and carry it around for later
        slope = 10 ** (learner.LearnerNN.order_of_magnitude(Vdot.detach().abs().max()))
        #relu = torch.nn.LeakyReLU(1 / slope.item())
        relu = torch.nn.ReLU()
        # relu = torch.nn.Softplus()
        # compute loss function. if last layer of ones (llo), can drop parts with V
        state_loss = -V_I
        goal_loss = V_G
        margin = 1e-5
        req_diff = (V_I.max()-V_G.min())/self.T
        lie_loss = relu(Vdot+req_diff)
        # Vdot never gets negative...

        subgrad = not convex
        
        if subgrad:
            supp_max = torch.tensor([-1.])
            lie_max = lie_loss.max()
            ind_lie_max = lie_loss.argmax()
            loss = lie_max
            sub_sample = -1
            for i, elem in enumerate(indices["lie"]):
                if ind_lie_max in elem:
                    sub_sample = i
                    break
            for ind in supp_samples:
                inds = indices["lie"][ind]
                supp_max = torch.max(supp_max, lie_loss[inds].max())
            supp_loss = supp_max
            new_sub_samples = set([sub_sample])
        else:
            loss = 0
            for inds in indices["lie"]:
                curr_max = torch.tensor(0.)
                if self.llo:
                    state_elems = torch.tensor(0.)
                else:
                    state_elems = state_loss[inds]
                lie_elems = lie_loss[inds]
                loss += lie_elems.max()
            #if self.llo:
            #    learn_accuracy = (Vdot <= -margin).count_nonzero().item()
            #    #loss = (relu(Vdot + margin * circle)).max()
            #    loss = (relu(Vdot + margin)).max()
            #else:
            #    learn_accuracy = 0.5 * (
            #        (Vdot <= -margin).count_nonzero().item()
            #        + (V >= margin).count_nonzero().item()
            #    )
            #    #loss = torch.max((relu(Vdot + margin * circle)).max() , (
            #    #    relu(-V + margin * circle)
            #    #).max()) # Why times circle?
            #    loss = torch.max((relu(Vdot + margin )).max() , (
            #        relu(-V + margin)
            #    ).max()) # Why times circle?
        goal_accuracy = (V_G<0).count_nonzero().item()/len(V_G)
        dom_accuracy = (V_I>0).count_nonzero().item()/len(V_I)
        lie_accuracy = (Vdot <= -req_diff).count_nonzero().item()/len(Vdot)
        accuracy = {"goal_acc": goal_accuracy * 100, "domain_acc" : dom_accuracy*100, "lie_acc": lie_accuracy*100}
        gamma = .1 
        state_con = relu(state_loss+margin).mean()
        goal_con = relu(goal_loss+margin).mean()
        loss = loss+ gamma*(state_con+goal_con)
        if supp_loss != -1:
            supp_loss = supp_loss + gamma*(state_con+goal_con)
        #try:
        #    final_ind = [ind for ind in indices["lie"] if len(ind) > 0][-1][-1]
        #except IndexError:
        #    final_ind = -1
        #if final_ind < len(lie_loss) - 1:
        #    loss += relu(lie_loss)[final_ind+1:].sum() + relu(state_loss)[final_ind+1:].sum
        #    if supp_loss != -1:
        #        supp_loss += relu(lie_loss)[final_ind+1:].sum() + relu(state_loss)[final_ind+1:].sum()


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

        samples = torch.cat([S[XD], S[XI], S[XG]])

        if type(Sdot[XG]) is list:
            idot3 = 0
        else:
            idot3 = Sdot[XG].shape[0]
        samples_dot = Sdot[XD]

        samples_with_nexts = samples[:idot1]
        states_only = torch.cat([samples[idot1:i1], samples[i1+idot2:i1+i2], samples[i1+i2+idot3:]])
        times = times[XD]

        supp_samples = set()
        for t in range(learn_loops):
            optimizer.zero_grad()
            if self.control:
                samples_dot = f_torch(samples)

            V1, Vdot, circle = learner.get_all(samples_with_nexts, samples_dot, times) # error here after discarding
            V2 = learner(states_only)
            #V = torch.cat([V1,V2])
            V = V2
            V_D = V[:i1-idot1]
            V_I = V[i1-idot1:i1+i2-i1-idot2]
            V_G = V[i1+i2-idot1-idot2:]


            loss, supp_loss, learn_accuracy, sub_sample = self.compute_loss(V_I, V_G, Vdot, Sind, supp_samples, convex)
            if loss <= best_loss:
                best_loss = loss
                best_net = copy.deepcopy(learner)

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
                    #print(inner)
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
        V2 = learner(states_only)
        #V = torch.cat([V1,V2])
        V = V2
        V_D = V[:i1-idot1]
        V_I = V[i1-idot1:i1+i2-i1-idot2]
        V_G = V[i1+i2-idot1-idot2:]



        loss, supp_loss, learn_accuracy, sub_sample = self.compute_loss(V_I, V_G, Vdot, Sind, supp_samples, convex)

        if self.control:
            loss = loss + control.cosine_reg(samples, samples_dot)
        if loss <= best_loss:
            best_loss = loss
            best_net = copy.deepcopy(learner)
        return {ScenAppStateKeys.loss: loss, "best_loss":best_loss, "best_net":best_net, "new_supps": supp_samples}

    def get_violations(self, V, Vdot, S, Sdot, times, state_data):
        req_diff = (V(state_data["init"]).max()-V(state_data["goal"]).min())/self.T
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
            if np.linalg.norm(traj[:, -1]) > 0.01: # check this does what I want it to do...
                true_violated += 1
            #if any(pred_V < pred_0):
            #    raise ValueError("Value violation!")
            if any(pred_Vdot[non_goal_inds] > -req_diff):
                violated += 1
        return violated, true_violated


    def get_constraints(self, verifier, V, Vdot) -> Generator:
        """
        :param verifier: verifier object
        :param V: SMT formula of Lyapunov Function
        :param Vdot: SMT formula of Lyapunov lie derivative
        :return: tuple of dictionaries of lyapunov conditons
        """
        _Or = verifier.solver_fncts()["Or"]
        _And = verifier.solver_fncts()["And"]
        _Not = verifier.solver_fncts()["Not"]

        if self.llo:
            # V is positive definite by construction
            lyap_negated = Vdot >= 0
        else:
            lyap_negated = _Or(V <= 0, Vdot >= 0)

        not_origin = _Not(_And(*[xi == 0 for xi in verifier.xs]))
        lyap_negated = _And(lyap_negated, not_origin)
        lyap_condition = _And(self.domain, lyap_negated)
        for cs in ({XD: lyap_condition},):
            yield cs


    def estimate_beta(self, net):
        # This function is unused I think
        print("Estimating beta!")
        try:
            border_D = self.D[XD].sample_border(300)
            beta, _ = net.compute_minimum(border_D)
        except NotImplementedError:
            beta = self.D[XD].generate_data(300)
        return beta

    @staticmethod
    def _assert_state(domains, data):
        domain_labels = set(domains.keys())
        data_labels = set(data.keys())
        _set_assertion(set([XD]), domain_labels, "Symbolic Domains")
        _set_assertion(set([XD]), data_labels, "Data Sets")

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
        slope = 1e-2  # (learner.orderOfMagnitude(max(abs(Vdot)).detach()))
        #relu6 = torch.nn.ReLU6()
        #splu = torch.nn.Softplus(beta=0.5)
        # init_loss = (torch.relu(B_i + margin) - slope * relu6(-B_i + margin)).mean()
        relu = torch.nn.ReLU()

        req_diff = (B_u.min() - B_i.max())/self.T
        lie_margin = 1e-3
        lie_loss = relu(Bdot_d-req_diff)
        # For convex this causes problems, need to double check it is still useful for non-convex now I've made some changes
        lie_accuracy = (
            100 * ((Bdot_d < 0).count_nonzero()).item() / Bdot_d.shape[0]
        )
        subgrad = not convex
        # subgradient descent

        if subgrad:
            supp_max = torch.tensor([-1.0])
            lie_max = lie_loss.max() # Setting this to 1000 helps the DT converge for some reason...
            ind_lie_max = lie_loss.argmax()
            loss = lie_max
            sub_sample = -1
            for i, elem in enumerate(indices["lie"]):
                if ind_lie_max in elem:
                    sub_sample = i
                    break
            for ind in supp_samples:
                lie_inds = indices["lie"][ind]
                if len(lie_inds) > 0:
                    supp_max = torch.max(supp_max, lie_loss[lie_inds].max())
                #supp_max = torch.max(supp_max, torch.max(torch.max(lie_loss[lie_inds].max(), unsafe_loss[unsafe_inds].max()), init_loss[init_inds].max()))
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

        
        # this bit adds on the additional samples of just states

        # TO FIX: if we have some samples with e.g. unsafe, but final sample does not have any then we assume there aren't any unsafe samples
        # Don't worry, state data is all added to the end of the data
        #import pdb; pdb.set_trace()
        
        # unsafe_loss = (torch.relu(-B_u + margin) - slope * relu6(B_u + margin)).mean()
        gamma = 0.1 
        unsafe_margin = 1e-3
        init_loss = gamma*(relu(B_i).mean())
        loss = loss+init_loss
        unsafe_loss = gamma*relu(-B_u+unsafe_margin).mean()
        loss = loss+ unsafe_loss
        if supp_loss != -1:
            supp_loss = supp_loss + init_loss
            supp_loss = supp_loss + unsafe_loss
        
        

        #loss = torch.max(torch.max(init_loss, unsafe_loss), lie_loss)

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
        # samples = torch.cat([s for s in S.values()])
        label_order = [XD, XI, XU]
        samples = torch.cat([S[label] for label in label_order if type(S[label]) is not list])
        samples_with_nexts = torch.cat([samples[:idot1], samples[i1:i1+idot2], samples[i1+i2:i1+i2+idot3]])
        states_only = torch.cat([samples[idot1:i1], samples[i1+idot2:i1+i2], samples[i1+i2+idot3:]])
        times = torch.cat([times[label] for label in label_order if type(times[label]) is not list])
        # samples_dot = torch.cat([s for s in Sdot.values()])
        samples_dot = torch.cat([Sdot[label] for label in label_order if type(Sdot[label]) is not list])
        supp_samples = set()
        for t in range(learn_loops):
            optimizer.zero_grad()


            # permutation_index = torch.randperm(S[0].size()[0])
            # permuted_S, permuted_Sdot = S[0][permutation_index], S_dot[0][permutation_index]
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
            # loss = loss + (100-percent_accuracy)

            if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                log_loss_acc(t, loss, accuracy, learner.verbose)

            # if learn_accuracy / batch_size > 0.99:
            #     for k in range(batch_size):
            #         if Vdot[k] > -margin:
            #             print("Vdot" + str(S[k].tolist()) + " = " + str(Vdot[k].tolist()))
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
                    #print(inner)
                    if inner <= 0:
                        supp_samples = supp_samples.union(sub_sample)
                        optimizer.zero_grad()
                        loss.backward()
                else:
                    supp_samples = supp_samples.union(sub_sample)
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
            if (any(pred_B_i >= 0) or
                    any(pred_B_u <= 0)):
                raise ValueError("Value violation!")
            if any(pred_B_dots > req_diff):
                violated += 1
                continue
        return violated, true_violated

    @staticmethod
    def _assert_state(domains, data):
        domain_labels = set(domains.keys())
        data_labels = set(data.keys())
        _set_assertion(set([XD, XI, XU]), domain_labels, "Symbolic Domains")
        _set_assertion(set([XD, XI, XU]), data_labels, "Data Sets")


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

    def compute_loss(self, V_i, V_u, V_d, V_g, Vdot_d, indices, supp_samples, convex):
        margin = 1e-3
        margin_lie = 0.0
        learn_accuracy = (V_i <= -margin).count_nonzero().item() + (
            V_u >= margin
        ).count_nonzero().item()
        percent_accuracy_init_unsafe = learn_accuracy * 100 / (len(V_i) + len(V_u))
        slope = 0  # 1 / 10**4  # (learner.orderOfMagnitude(max(abs(Vdot)).detach()))
        relu = torch.nn.ReLU()

        subgrad = not convex

        
        lie_index = torch.nonzero(V_d < -margin)

        if lie_index.nelement() != 0:
            init_loss = relu(V_i + margin).mean()
            unsafe_loss = relu(-V_u + margin).mean()
            req_diff = (V_i.max()-V_g.min())/self.T
            # this might need changing in case there are points in the unsafe or goal set?
            # ensure no goal states in domain data (see rwa_2 for example)
            Vdot_selected = torch.index_select(Vdot_d, dim=0, index=lie_index[:, 0])
            lie_loss = relu(Vdot_selected+req_diff)*10
            lie_accuracy=(((Vdot_selected <= -req_diff).count_nonzero()).item() * 100 / Vdot_selected.shape[0]
            )
            if subgrad:
                supp_max = torch.tensor([-1.0])
                lie_max = lie_loss.max()
                ind_lie_max = lie_loss.argmax()
                loss = lie_max
                sub_sample = -1
                for i, elem in enumerate(indices["lie"]):
                    if lie_index[ind_lie_max,0] in elem:
                        sub_sample = i
                        break
                for ind in supp_samples:
                    lie_inds = indices["lie"][ind]
                    adjusted_inds = torch.cat([torch.where(lie_index[:,0] == elem)[0] for elem in lie_inds])
                    if len(adjusted_inds) > 0:
                        supp_max = torch.max(supp_max, lie_loss[adjusted_inds].max())
                supp_loss = supp_max
                new_sub_samples = set([sub_sample])
                loss = loss + init_loss + unsafe_loss
                if supp_loss != -1:
                    supp_loss = supp_loss + init_loss + unsafe_loss
            else:
                raise NotImplementedError
        else:
            # If this set is empty then the function is not negative enough across XS, so only penalise the initial set
            lie_accuracy = 0.0
            loss = (V_i + margin).mean()
            supp_loss = -1
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
        label_order = [XD, XI, XG, XS_BORDER]
        samples = torch.cat([S[label] for label in label_order if type(S[label]) is not list])
        # samples = torch.cat((S[XD], S[XI], S[XU]))

        samples_dot = Sdot[XD]
        samples_with_nexts = S[XD][:idot1]
        #samples_dot = torch.cat([Sdot[label] for label in label_order if type(Sdot[label]) is not list])
        #samples_with_nexts = torch.cat([samples[:idot1], samples[i1:i1+idot2], samples[i1+i2:i1+i2+idot3], samples[i1+i2+i3:i1+i2+i3+idot4]])
        states_only = torch.cat([samples[idot1:i1], samples[i1+idot2:i1+i2], samples[i1+i2+idot3:i1+i2+i3], samples[i1+i2+i3+idot4:]])
        times = torch.cat([times[label] for label in label_order if type(times[label]) is not list])
        supp_samples = set()

        for t in range(learn_loops):
            optimizer.zero_grad()

            B_d, Bdot_d, _ = learner.get_all(samples_with_nexts, samples_dot, times[:idot1])

            #nn, grad_nn = learner.compute_net_gradnet(samples)

            #V, gradV = learner.compute_V_gradV(nn, grad_nn, samples)
            B = learner(states_only)
            
            B_i = B[i1-idot1 : i1 + i2-idot1-idot2]
            B_g = B[i1 + i2-idot1-idot2 :i1+i2+i3-idot1-idot2-idot3]
            B_u = B[i1+i2+i3-idot1-idot2-idot3:]
            

            loss, supp_loss, accuracy, sub_sample = self.compute_loss(B_i, B_u, B_d, B_g, Bdot_d, Sind, supp_samples, convex)

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
                    #print(inner)
                    if inner <= 0:
                        supp_samples = supp_samples.union(sub_sample)
                        optimizer.zero_grad()
                        loss.backward()
                else:
                    supp_samples = supp_samples.union(sub_sample)
            optimizer.step()
        B_d, Bdot_d, _ = learner.get_all(samples_with_nexts, samples_dot, times[:idot1])
        B_i = B[i1-idot1:i1+i2-idot2-idot1]
        B_g = B[i1 + i2-idot1-idot2 :i1+i2+i3-idot1-idot2-idot3]
        B_u = B[i1+i2+i3-idot1-idot2-idot3:]
        loss, supp_loss, accuracy, sub_sample = self.compute_loss(B_i, B_u, B_d, B_g, Bdot_d, Sind, supp_samples, convex)
        
        if loss <= best_loss:
            best_loss = loss
            best_net = copy.deepcopy(learner)
            print(supp_samples)
        return {ScenAppStateKeys.loss: loss, "best_loss":best_loss, "best_net":best_net, "new_supps": supp_samples}

    def get_violations(self, B, Bdot, S, Sdot, times, states):
        req_diff = (B(states["init"]).max()-B(states["goal"]).min())/self.T
        true_violated = 0
        violated = 0
        for i, (traj, traj_deriv, time) in enumerate(zip(S, Sdot, times)):
            traj, traj_deriv, time = torch.tensor(traj.T, dtype=torch.float32), torch.tensor(np.array(traj_deriv).T, dtype=torch.float32), torch.tensor(time, dtype=torch.float32)

            valid_inds = torch.where(self.D[XD].check_containment(traj))
            
            traj = traj[valid_inds]
            traj_deriv = traj_deriv[valid_inds]
            time = time[valid_inds]

            initial_inds = torch.where(self.D[XI].check_containment(traj))
            
            # getting too many violations, need to investigate

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

    def compute_beta_loss(self, beta, V_g, Vdot_g, V_d, indices, supp_samples, convex):
        """Compute the loss for the beta condition
        :param beta: the guess value of beta based on the min of V of XG_border
        :param V_d: the value of V at points in the goal set
        :param Vdot_d: the value of the lie derivative of V at points in the goal set"""
        lie_index = torch.nonzero(V_g <= beta)
        
        relu = torch.nn.ReLU()
        margin = 1e-5

        if lie_index.nelement() != 0:
            subgrad = not convex
            beta_lie = relu(torch.index_select(Vdot_g, dim=0, index=lie_index[:, 0]) + margin)
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

            V_i = Vstates[init_indices[0] : init_indices[1]]
            V_u = Vstates[unsafe_indices[0] : unsafe_indices[1]]
            S_dg = samples[goal_border_indices[0] : goal_border_indices[1]]
            V_g = Vstates[goal_indices[0] : goal_indices[1]]
            
            Vdot_g = Vdot[lie_dot_indices[1] :]
            samples_dot_d = samples_dot[: lie_indices[1]]

            loss, supp_loss, accuracy, sub_sample  = self.compute_loss(V_i, V_u, V_d, V_g, gradV_d, Sind, supp_samples, convex)

            beta = learner.compute_minimum(S_dg)[0]
            #beta_loss, supp_beta_loss, beta_sub_sample = 0, -1, set()
            # converges without beta loss
            beta_loss, supp_beta_loss, beta_sub_sample = self.compute_beta_loss(beta, V_g, Vdot_g, V_d, Sind, supp_samples, convex)
            loss = torch.max(loss,beta_loss)
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
                else:
                    supp_samples = supp_samples.union(sub_sample)
            optimizer.step()
        V, Vdot, _ = learner.get_all(samples_with_nexts, samples_dot, torch.cat([times[:lie_dot_indices[1]],times[-len(Sdot[XG]):]]))

        gradV_d = Vdot[: lie_dot_indices[1]]

        Vstates = learner(samples)
        
        V_d = Vstates[lie_indices[0] : lie_indices[1]]
        V_i = Vstates[init_indices[0] : init_indices[1]]
        V_u = Vstates[unsafe_indices[0] : unsafe_indices[1]]
        S_dg = samples[goal_border_indices[0] : goal_border_indices[1]]
        V_g = Vstates[goal_indices[0] : goal_indices[1]]
        
        Vdot_g = Vdot[lie_dot_indices[1] :]
        samples_dot_d = samples_dot[: lie_indices[1]]
        
        loss, supp_loss, accuracy, sub_sample  = self.compute_loss(V_i, V_u, V_d, V_g, gradV_d, Sind, supp_samples, convex)

        beta = learner.compute_minimum(S_dg)[0]
        beta_loss, supp_beta_loss, beta_sub_sample = self.compute_beta_loss(beta, V_g, Vdot_g, V_d, Sind, supp_samples, convex)
        loss = torch.max(loss, beta_loss)

        if loss <= best_loss:
            best_loss = loss
            best_net = copy.deepcopy(learner)

        return {ScenAppStateKeys.loss: loss, "best_loss":best_loss, "best_net":best_net, "new_supps":supp_samples}


    def beta_search(self, learner, verifier, C, Cdot, S):
        import pdb; pdb.set_trace()
        return learner.compute_minimum(S_dg)[0]
            


class SafeROA(Certificate):
    """Certificate to prove stable while safe"""

    def __init__(self, domains, config: ScenAppConfig) -> None:
        self.ROA = ROA(domains, config)
        self.barrier = Barrier._for_safe_roa(domains, config)
        self.bias = self.ROA.bias, self.barrier.bias
        self.beta = None
        self.D = config.DOMAINS
        self.T = config.SYSTEM.time_horizon

    def learn(
        self,
        learner: tuple[learner.LearnerNN, learner.LearnerNN],
        optimizer: Optimizer,
        S: dict,
        Sdot: dict,
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
        lyap_learner = learner[0]
        barrier_learner = learner[1]

        learn_loops = 1000
        lie_indices = 0, S[XD].shape[0]
        
        if XR in S.keys():
            # The idea here is that the data set for barrier learning is not conducive to learning the region of attraction (which should ideally only contain stable points that converge.
            # So we allow for a backup data set used only for the ROA learning. If not passed, we use the same data set as for the barrier learning.
            r_indices = lie_indices[1], lie_indices[1] + S[XR].shape[0]
        else:
            r_indices = lie_indices[0], lie_indices[1]
        
        init_indices = r_indices[1], r_indices[1] + S[XI].shape[0]
        unsafe_indices = init_indices[1], init_indices[1] + S[XU].shape[0]
        label_order = [XD, XR, XI, XU]
        samples = torch.cat([S[label] for label in label_order if label in S])
        samples = torch.cat([s for s in S.values()])

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            samples_dot = torch.cat([s for s in Sdot.values()])

        for t in range(learn_loops):
            optimizer.zero_grad()

            if f_torch:
                samples_dot = f_torch(samples)

            # This seems slightly faster
            V, Vdot, circle = lyap_learner.get_all(
                samples[r_indices[0] : r_indices[1]],
                samples_dot[r_indices[0] : r_indices[1]],
            )
            B, Bdot, _ = barrier_learner.get_all(samples, samples_dot)
            (
                B_d,
                Bdot_d,
            ) = (
                B[lie_indices[0] : lie_indices[1]],
                Bdot[lie_indices[0] : lie_indices[1]],
            )
            B_i = B[init_indices[0] : init_indices[1]]
            B_u = B[unsafe_indices[0] : unsafe_indices[1]]

            lyap_loss, lyap_acc = self.ROA.compute_loss(V, Vdot, circle)
            b_loss, barr_acc = self.barrier.compute_loss(B_i, B_u, B_d, Bdot_d)

            loss = lyap_loss + b_loss

            accuracy = {**lyap_acc, **barr_acc}

            if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                log_loss_acc(t, loss, accuracy, lyap_learner.verbose)

            if (
                t > 1
                and accuracy["acc"] == 100
                and accuracy["acc init unsafe"] == 100
                and accuracy["acc belt"] >= 99.9
            ):
                break

            loss.backward()
            optimizer.step()

        SI = S[XI]
        self.ROA.beta = lyap_learner.compute_maximum(SI)[0]
        lyap_learner.beta = self.ROA.beta

        return {}


class ReachAvoidRemain(Certificate):
    def __init__(self, domains, config: ScenAppConfig) -> None:
        self.domains = domains
        self.RWS = RWS(domains, config)
        self.barrier = Barrier._for_goal_final(domains, config)
        self.BORDERS = (XS,)
        self.bias = self.RWS.bias, self.barrier.bias
        self.D = config.DOMAINS
        self.T = config.SYSTEM.time_horizon
        

    def learn(
        self,
        learner: tuple[learner.LearnerNN, learner.LearnerNN],
        optimizer: Optimizer,
        S: dict,
        Sdot: dict,
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
        rws_learner = learner[0]  # lyap_learner
        barrier_learner = learner[1]  # barrier_learner

        learn_loops = 1000
        condition_old = False
        lie_indices = 0, S[XD].shape[0]
        init_indices = lie_indices[1], lie_indices[1] + S[XI].shape[0]
        unsafe_indices = init_indices[1], init_indices[1] + S[XU].shape[0]
        goal_indices = (
            unsafe_indices[1],
            unsafe_indices[1] + S[XG].shape[0],
        )

        final_indices = goal_indices[1], goal_indices[1] + S[XF].shape[0]
        nonfinal_indices = final_indices[1], final_indices[1] + S[XNF].shape[0]

        label_order = [XD, XI, XU, XG, XF, XNF]
        samples = torch.cat([S[label] for label in label_order])

        if f_torch:
            samples_dot = f_torch(samples)
        else:
            samples_dot = torch.cat([s for s in Sdot.values()])

        for t in range(learn_loops):
            optimizer.zero_grad()

            if f_torch:
                samples_dot = f_torch(samples)

            # This is messy
            nn, grad_nn = rws_learner.compute_net_gradnet(samples)

            V, gradV = rws_learner.compute_V_gradV(nn, grad_nn, samples)
            V_i = V[init_indices[0] : init_indices[1]]
            V_u = V[unsafe_indices[0] : unsafe_indices[1]]
            V_d = V[lie_indices[0] : lie_indices[1]]
            gradV_d = gradV[lie_indices[0] : lie_indices[1]]
            samples_dot_d = samples_dot[lie_indices[0] : lie_indices[1]]

            rws_loss, rws_acc = self.RWS.compute_loss(
                V_i, V_u, V_d, gradV_d, samples_dot_d
            )

            B, Bdot, _ = barrier_learner.get_all(samples, samples_dot)
            B_i = B[goal_indices[0] : goal_indices[1]]
            B_u = B[nonfinal_indices[0] : nonfinal_indices[1]]

            # Ideally the final set is very similar to the goal set, so sometimes the belt set is empty
            # as B is negative over it. So lets use data from the goal and nonfinal sets too (this seems to work well)
            B_d = B[goal_indices[0] : nonfinal_indices[1]]
            Bdot_d = Bdot[goal_indices[0] : nonfinal_indices[1]]
            b_loss, barr_acc = self.barrier.compute_loss(B_i, B_u, B_d, Bdot_d)

            loss = rws_loss + b_loss

            if f_torch:
                S_d = samples[: lie_indices[1]]
                loss = loss + control.cosine_reg(S_d, samples_dot_d)

            barr_acc["acc goal final"] = barr_acc.pop("acc init unsafe")

            accuracy = {**rws_acc, **barr_acc}

            if t % int(learn_loops / 10) == 0 or learn_loops - t < 10:
                log_loss_acc(t, loss, accuracy, rws_learner.verbose)

            if (
                accuracy["acc init unsafe"] == 100
                and accuracy["acc lie"] >= 100
                and accuracy["acc goal final"] >= 100
                and accuracy["acc belt"] >= 99.9
            ):
                condition = True
            else:
                condition = False

            if condition and condition_old:
                break
            condition_old = condition

            loss.backward()
            optimizer.step()

        return {}


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
        if self.certificate == CertificateType.LYAPUNOV:
            return self.auto_lyap()
        elif self.certificate == CertificateType.PRACTICALLYAPUNOV:
            return self.auto_practical_lyap()
        elif self.certificate == CertificateType.BARRIERALT:
            self.auto_barrier_alt(self.sets)
        elif self.certificate == CertificateType.RWS:
            self.auto_rws(self.sets)
        elif self.certificate == CertificateType.RSWS:
            self.auto_rsws(self.sets)
        elif self.certificate == CertificateType.STABLESAFE:
            self.auto_stablesafe(self.sets)
        elif self.certificate == CertificateType.RAR:
            self.auto_rar(self.sets)

    def auto_lyap(self) -> None:
        domains = {XD: self.XD}
        data = {XD: self.XD._generate_data(1000)}
        return domains, data


def get_certificate(
    certificate: CertificateType, custom_cert=None
) -> Type[Certificate]:
    if certificate == CertificateType.LYAPUNOV:
        return Lyapunov
    elif certificate == CertificateType.PRACTICALLYAPUNOV:
        return Practical_Lyapunov
    elif certificate == CertificateType.BARRIERALT:
        return BarrierAlt
    elif certificate in (CertificateType.RWS, CertificateType.RWA):
        return RWS
    elif certificate in (CertificateType.RSWS, CertificateType.RSWA):
        return RSWS
    elif certificate == CertificateType.STABLESAFE:
        return SafeROA
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
