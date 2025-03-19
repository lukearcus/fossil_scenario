from typing import NamedTuple, Union

import fossil.learner as learner
import fossil.verifier as verifier
from fossil.consts import * 
import fossil.consolidator as consolidator
import fossil.logger as logger
import fossil.certificate as certificate

from itertools import chain
from time import perf_counter, clock_gettime

import torch
import copy
import sympy as sp
from scipy import stats

from scipy.stats import beta as betaF

scenapp_log = logger.Logger.setup_logger(__name__)

class Stats(NamedTuple):
    iters: int
    N_data: int
    times: dict
    seed: int

class Result(NamedTuple):
    res: float 
    a_post_res: float
    cert: learner.LearnerNN
    stats: Stats


class SingleScenApp:
    def __init__(self, config: ScenAppConfig):
        self.config = config
        
        self.x, self.x_map, self.domains = self._initialise_domains()
        self.S, self.S_traj = self._initialise_data(self.config.DATA["full_data"], self.config.DATA["states_only"]) # Needs editing
        self.certificate = self._initialise_certificate()
        self.learner = self._initialise_learner()
        if config.CONVEX_NET:
            self.a_priori_supps = sum([param.numel() for param in self.learner.parameters()]) # Take this and add any violations for convex
        else:
            self.a_priori_supps = None
        self.verifier = self._initialise_verifier() 
        self.optimizer = self._initialise_optimizer() 
        if self.config.VERBOSE:
            logger.Logger.set_logger_level(self.config.VERBOSE)
    
    def _initialise_domains(self):
        x = verifier.get_verifier_type(self.config.VERIFIER).new_vars(
            self.config.N_VARS
            )
        x_map = {str(x): x for x in x}
        domains = {
                    label: domain.generate_boundary(x)
                    if label in certificate.BORDERS
                    else domain.generate_domain(x)
                    for label, domain in self.config.DOMAINS.items()
                  }
        if self.config.CERTIFICATE == CertificateType.RAR:
            domains[certificate.XNF] = self.config.DOMAINS[
                                        certificate.XF
                                                ].generate_complement(x)

        scenapp_log.debug("Domains: {}".format(domains))
        return x, x_map, domains


    def _initialise_certificate(self):
        custom_certificate = self.config.CUSTOM_CERTIFICATE
        certificate_type = certificate.get_certificate(self.config.CERTIFICATE, custom_certificate)
        if self.config.CERTIFICATE == certificate.CertificateType.STABLESAFE:
            raise ValueError("StableSafe not compatible with default CEGIS")
        return certificate_type(self.domains, self.config)


    def _initialise_learner(self):
        learner_type = learner.get_learner(
                    self.config.TIME_DOMAIN, self.config.CTRLAYER
                        )
        learner_instance = learner_type(
            self.config.N_VARS,
            self.certificate.learn,
            *self.config.N_HIDDEN_NEURONS,
            activation=self.config.ACTIVATION,
            bias=self.certificate.bias,
            config=self.config,
                            )
        return learner_instance

    def _initialise_verifier(self):
        num_params = sum(p.numel() for p in self.learner.parameters() if p.requires_grad)

        verifier_type = verifier.get_verifier_type(self.config.VERIFIER)
        verifier_instance = verifier_type(
                    self.config.N_VARS,
                    self.config.BETA,
                    self.config.N_DATA,
                    num_params,
                    self.config.VERBOSE,
                            )
        return verifier_instance


    def _initialise_data(self, traj_data, state_data):
        lumped_data = {key: torch.tensor(np.hstack(traj_data[key]), dtype=torch.float32 ) for key in traj_data} 

        traj_inds = [] 
        curr_ind = 0
        for elem in traj_data["times"]:
            if type(elem) is not np.float64:
                elem_len = len(elem)
            else:
                elem_len = 1
            traj_inds.append((curr_ind, curr_ind+elem_len))
            curr_ind += elem_len

        domained_data = {"states":{},"times":{},"derivs":{}, "indices":{}}
        for key in self.config.DOMAINS:
            domain = self.config.DOMAINS[key]
            domained_data["states"][key] = []
            domained_data["derivs"][key] = []
            domained_data["times"][key] = []
            domained_data["indices"][key] = [[] for elem in traj_inds]
            curr_ind = 0
            for ind, elem in enumerate(lumped_data["states"].T):
                if domain.check_containment(elem.expand([1,elem.size(dim=0)])):
                    for i, index in enumerate(traj_inds):
                        if ind in range(*index):
                            sample_ind = i
                            break
                    domained_data["indices"][key][sample_ind].append(curr_ind)
                    curr_ind += 1
                    domained_data["states"][key].append(lumped_data["states"][:,ind])
                    domained_data["derivs"][key].append(lumped_data["derivs"][:,ind])
                    domained_data["times"][key].append(lumped_data["times"][ind])
                    
            if len(domained_data["states"][key]) > 0:
                if key in state_data:
                    domained_data["states"][key] = torch.cat((torch.stack(domained_data["states"][key]), state_data[key]))
                else:
                    domained_data["states"][key] = torch.stack(domained_data["states"][key])
                domained_data["derivs"][key] = torch.stack(domained_data["derivs"][key])
                domained_data["times"][key] = torch.stack(domained_data["times"][key])
            else:
                domained_data["states"][key] = state_data[key]
        
        return domained_data, traj_data


    def _initialise_optimizer(self):
        #return torch.optim.SGD(
        return torch.optim.AdamW(
                [{"params": self.learner.parameters()}], # Might need to change this to consider controller parameters
                lr=self.config.LEARNING_RATE,
                )

    def update_controller(self, state):
        scenapp_log.debug("Updating controller does nothing")
        return state 

    def a_post_verify(self, cert, cert_deriv, n_data):
        state_data = self.config.DATA["states_only"]
        torch.manual_seed(clock_gettime(0))      #allows different samples when running in parallel
        try:
            test_data = self.config.DOMAINS["init"]._generate_data(n_data)()
        except KeyError:
            test_data = self.config.DOMAINS["lie"]._generate_data(n_data)()

        all_test_data = self.config.SYSTEM().generate_trajs(test_data)
        data = {"states_only": None, "full_data": {"times":all_test_data[0],"states":all_test_data[1],"derivs":all_test_data[2]}}
        num_violations, true_violations = self.certificate.get_violations(cert, cert_deriv, data["full_data"]["states"], data["full_data"]["derivs"], data["full_data"]["times"], state_data)
        k = num_violations
        k = true_violations # use this for direct property validation
        beta_bar = self.config.BETA[0]
        N = n_data
        d = 1
        eps = betaF.ppf(1-beta_bar, k+d, N-(d+k)+1) 
        print("Direct Property scenario approach risk: {:.5f}".format(eps))
        print("Certificate violation rate: {:.3f}".format(num_violations/n_data))
        print("Property violation rate: {:.3f}".format(true_violations/n_data))
        return eps


    def discard(self, state):
        
        # Discard all samples that were of support for last run...
        # Could discard just the current worst case for better guarantees but worse performance
        # Could probably do this by just discarding last support sample...
        
        traj_data = self.config.DATA["full_data"]
        if not self.config.CONVEX_NET:
            if len(state["discarded"]) == 0:
                state["discarded"] = state["supps"]
                self.remaining_inds = list(set(range(len(traj_data["states"])))-state["discarded"])
            else:
                to_remove = set()
                for new_disc in state["supps"]:
                    actual_ind = self.remaining_inds[new_disc]
                    state["discarded"].add(actual_ind)
                    to_remove.add(actual_ind)
                if len(to_remove) == len(self.remaining_inds):
                    print("removed all samples, maintaining final support samples")
                    return
                self.remaining_inds=list(set(self.remaining_inds)-to_remove)
            state["supps"] = set()
        new_traj_inds = [i for i in range(len(traj_data["states"])) if i not in state["discarded"]]
        new_traj_data = {}
        for key in traj_data:
            new_traj_data[key] = [traj_data[key][ind] for ind in new_traj_inds]
        
        self.S, self.S_traj = self._initialise_data(new_traj_data, self.config.DATA["states_only"]) # Needs editing
        
        state[ScenAppStateKeys.S] = self.S["states"]
        state[ScenAppStateKeys.S_dot] = self.S["derivs"]
        state[ScenAppStateKeys.S_traj] = self.S_traj["states"]
        state[ScenAppStateKeys.S_traj_dot] =  self.S_traj["derivs"]
        state[ScenAppStateKeys.S_inds] =  self.S["indices"]
        state[ScenAppStateKeys.times] = self.S["times"]
        return

    def est_disc_gap(self, state):
        # Would be better off adding this to the loss function, but this works OK.
        # Adding to loss function would likely be quite slow...

        t_max = max([elem.max() for elem in state[ScenAppStateKeys.times].values() if type(elem) is not list])
        state_data = np.hstack(state[ScenAppStateKeys.S_traj])
        next_data = np.hstack(state[ScenAppStateKeys.S_traj_dot])
        times = np.hstack(self.S_traj["times"])

        valid_inds = torch.where(self.config.DOMAINS["lie"].check_containment(torch.Tensor(state_data.T)))
        state_data = state_data[:,valid_inds[0]]
        next_data = next_data[:,valid_inds[0]]
        times = times[valid_inds[0]]
        inds = np.arange(0, len(valid_inds[0]))

        M_f = 0
        M_v = 0
    
        M = 10
        N = 1000
        alpha = 0.1
        psi_f = []
        psi_v = []
        for i in range(M):
            max_s_f = 0
            max_s_v = 0
            for j in range(N):
                poss_inds = [[]]
                while len(poss_inds[0]) <= 1:
                    ind  =np.random.choice(inds)
                    x = state_data[:,[ind]]
                    poss_inds = np.where(np.linalg.norm(state_data-state_data[:,[ind]],axis=0)<alpha)
                y_ind = ind
                while y_ind == ind:
                    y_ind = np.random.choice(poss_inds[0])
                y = state_data[:,[y_ind]]
                _, grad = state[ScenAppStateKeys.best_net].compute_net_gradnet(torch.Tensor(np.hstack([x,y]).T))  
                s_v = np.linalg.norm(grad[0].detach().numpy()-grad[1].detach().numpy())/np.linalg.norm(x-y)
                
                x_tau = next_data[:, [ind]]
                y_tau = next_data[:, [y_ind]]
                tau = np.min([times[ind], times[y_ind]])
                if self.config.TIME_DOMAIN == TimeDomain.CONTINUOUS: 
                    s_f = np.linalg.norm((x_tau-x-y_tau+y)/tau)/np.linalg.norm(x-y)
                else:
                    s_f = np.linalg.norm(x_tau-y_tau)/np.linalg.norm(x-y)
                max_s_f = max(s_f,max_s_f)
                max_s_v = max(s_v,max_s_v)
                M_f = max(M_f, np.linalg.norm((y_tau-y)/times[y_ind]))
                M_f = max(M_f, np.linalg.norm((x_tau-x)/times[ind]))

                M_v = max(M_v, np.linalg.norm(grad[0].detach().numpy()))
                M_v = max(M_v, np.linalg.norm(grad[1].detach().numpy()))
            psi_f.append(-max_s_f)
            psi_v.append(-max_s_v)
        _, _, L_f, _ = stats.exponweib.fit(psi_f)
        L_f = -L_f
        _, _, L_v, _ = stats.exponweib.fit(psi_v)
        L_v = -L_v
        delta = (t_max)*M_f*(M_v*L_f+M_f*L_v)
        return delta

    def solve(self) -> Result:
        converge_tol = 1e-4
        Sdot = self.S["derivs"]
        S = self.S["states"]
        S_inds = self.S["indices"]
        S_traj = self.S_traj
        times = self.S["times"]
        # Initialize CEGIS state
        state = self.init_state(Sdot, S, S_traj, S_inds, times)

        # Reset timers for components
        self.learner.get_timer().reset()
        state["net_dot"] = self.learner.nn_dot
        iters = 0
        stop = False
        N_data = self.config.N_DATA
        n_test_data = self.config.N_TEST_DATA
        old_loss = float("Inf")
        old_best = float("Inf")
        if self.config.CONVEX_NET:
            state["supps"] = {"active":0,"relaxed":0}
        else:
            state["supps"] = set()
        state["supp_len"] = self.a_priori_supps
        while not stop:
            scenapp_log.debug("\033[1m Learner \033[0m")
            outputs = self.learner.get(**state)
            state = {**state, **outputs}
            
            if self.config.CONVEX_NET:
                state["supps"] = outputs["new_supps"]
            else:
                state["supps"] = state["supps"].union(outputs["new_supps"])
            state = self.update_controller(state)
            if self.config.CONVEX_NET and torch.abs(state["loss"]-old_loss) < converge_tol:
                scenapp_log.debug("\033[1m Verifier \033[0m")
                

                outputs = self.verifier.get(**state)
                state = {**state, **outputs}
                print("Epsilon: {:.5f}".format(state[ScenAppStateKeys.bounds]))
                stop = self.process_certificate(S, state, iters)

            elif not self.config.CONVEX_NET and state["best_loss"] <= 0.0:
                
                if self.config.CALC_DISC_GAP:
                    scenapp_log.debug("negative best loss")
                    delta = self.est_disc_gap(state)
                    if state["best_loss"] > - delta:
                        iters += 1
                        old_loss = state["loss"]
                        old_best = state["best_loss"]
                        scenapp_log.info("Required delta: {:.5f}".format(delta))
                        scenapp_log.info("Iteration: {}".format(iters))
                    else:
                        scenapp_log.info("Required delta: {:.5f}".format(delta))
                        scenapp_log.info("Best loss: {:.5f}".format(state["best_loss"]))
                        scenapp_log.debug("\033[1m Verifier \033[0m")
                        

                        outputs = self.verifier.get(**state)
                        state = {**state, **outputs}

                        print("Epsilon: {:.5f}".format(state[ScenAppStateKeys.bounds]))
                        stop = self.process_certificate(S, state, iters)

                else:
                    scenapp_log.debug("\033[1m Verifier \033[0m")
                    

                    outputs = self.verifier.get(**state)
                    state = {**state, **outputs}

                    print("Epsilon: {:.5f}".format(state[ScenAppStateKeys.bounds]))
                    stop = self.process_certificate(S, state, iters)
            
            elif state[ScenAppStateKeys.verification_timed_out]:
                scenapp_log.warning("Verification timed out")
                stop = True
                state[ScenAppStateKeys.bounds] = None
            elif (
                    self.config.SCENAPP_MAX_ITERS <= iters
                    ):
                scenapp_log.warning("Out of iterations")
                stop = True
                state[ScenAppStateKeys.bounds] = None
            elif not self.config.CONVEX_NET and torch.abs(state["best_loss"]-old_best) < converge_tol:
                scenapp_log.info("Convergence reached, but failed to find valid certificate, discarding samples")
                self.discard(state)
                scenapp_log.debug("Discarded {} samples so far".format(len(state["discarded"])))
                iters += 1
                old_loss = state["loss"]
                old_best = state["best_loss"]
                scenapp_log.info("Iteration: {}".format(iters))

            elif not (
                    state[ScenAppStateKeys.found]
                    or state[ScenAppStateKeys.verification_timed_out]
                    ):

                iters += 1
                old_loss = state["loss"]
                old_best = state["best_loss"]
                scenapp_log.info("Iteration: {}".format(iters))
            if type(old_best) is float:
                scenapp_log.info("Best loss: {:.10f}".format(old_best))
            else:
                scenapp_log.info("Best loss: {:.10f}".format(old_best.item()))
            scenapp_log.info("Current loss: {:.10f}".format(state["loss"]))
        state = self.process_timers(state)

        stats = Stats(
                iters, N_data, state["components_times"], torch.initial_seed()
                )
        pre_post = perf_counter()
        a_post_eps = self.a_post_verify(state[ScenAppStateKeys.best_net], state[ScenAppStateKeys.best_net].nn_dot, n_test_data)
        print("Direct property guarantee time: {:.5f}s".format(perf_counter()-pre_post))
        self._result = Result(state[ScenAppStateKeys.bounds], a_post_eps, state[ScenAppStateKeys.best_net], stats)
                #state[ScenAppStateKeys.net], state[ScenAppStateKeys.net_dot], n_test_data)
        return self._result

    def init_state(self, Sdot, S, S_traj, S_inds, times):
        state = {
                ScenAppStateKeys.net: self.learner,
                ScenAppStateKeys.optimizer: self.optimizer,
                ScenAppStateKeys.S: S,
                ScenAppStateKeys.S_dot: Sdot,
                ScenAppStateKeys.S_traj: S_traj["states"],
                ScenAppStateKeys.S_traj_dot: S_traj["derivs"],
                ScenAppStateKeys.S_inds: S_inds,
                ScenAppStateKeys.times: times,
                ScenAppStateKeys.V: None,
                ScenAppStateKeys.V_dot: None,
                ScenAppStateKeys.x_v_map: self.x_map,
                ScenAppStateKeys.found: False,
                ScenAppStateKeys.verification_timed_out: False,
                ScenAppStateKeys.trajectory: None,
                ScenAppStateKeys.ENet: self.config.ENET,
                ScenAppStateKeys.best_loss: np.inf,
                ScenAppStateKeys.best_net: None,
                ScenAppStateKeys.discarded: set(),
                ScenAppStateKeys.convex: self.config.CONVEX_NET,
                ScenAppStateKeys.discrete: self.config.TIME_DOMAIN != TimeDomain.CONTINUOUS,
                }

        return state

    def process_timers(self, state: dict[str, Any]) -> dict[str, Any]:
        state[ScenAppStateKeys.components_times] = [
                self.learner.get_timer().sum,
                self.verifier.get_timer().sum,
                ]
        print("Learner times: {}".format(self.learner.get_timer()))
        scenapp_log.info("Verifier times: {}".format(self.verifier.get_timer()))
        return state

    def process_certificate(
            self, S: dict[str, torch.Tensor], state: dict[str, Any], iters: int
            ) -> bool:
        stop = False
        if (
                self.config.CERTIFICATE == CertificateType.LYAPUNOV
                or self.config.CERTIFICATE == CertificateType.ROA
                ):
            self.learner.beta = self.certificate.estimate_beta(self.learner)

        #if isinstance(self.f, control.GeneralClosedLoopModel):
        #    raise NotImplementedError("Can't do controlled models")
        #    ctrl = " and controller"
        #else:
        ctrl = ""
        print(f"Found a valid {self.config.CERTIFICATE.name} certificate" + ctrl)
        stop = True
        return stop

    @property
    def result(self):
        return self._result

    def _assert_state(self):
        assert self.config.LEARNING_RATE > 0
        assert self.config.CEGIS_MAX_TIME_S > 0
        if self.config.TIME_DOMAIN == TimeDomain.DISCRETE:
            assert self.config.CERTIFICATE in (
                CertificateType.LYAPUNOV,
                CertificateType.BARRIERALT,
                )
                # Passing sets to Fossil is complicated atm and I've messed it up (passing too many can lead to bugs too).
                # This is a temporary debug check until some better way of passing sets is implemented.
            self.certificate._assert_state(self.domains, self.S)

class DoubleScenApp(SingleScenApp):
    # Not sure if this works currently

    def __init__(self, config: ScenAppConfig):
        super().__init__(config)
        self.lyap_learner, self.barr_learner = self.learner
    
    def _initialise_certificate(self):
        custom_certificate = self.config.CUSTOM_CERTIFICATE
        cert_type = certificate.get_certificate(self.config.CERTIFICATE, custom_certificate)
        if self.config.CERTIFICATE != CertificateType.RAR:
            raise ValueError("DoubleScenApp only suppots RAR certificates")
        return cert_type(self.domains, self.config)

    def _initialise_learner(self):
        learner_type = learner.get_learner(self.config.TIME_DOMAIN, self.config.CTRLAYER)

        lyap_learner = learner_type(
            self.config.N_VARS,
            self.certificate.learn,
            *self.config.N_HIDDEN_NEURONS,
            activation=self.config.ACTIVATION,
            bias=self.certificate.bias[0],
            config=self.config,
                            )
        
        barr_learner = learner_type(
            self.config.N_VARS,
            self.certificate.learn,
            *self.config.N_HIDDEN_NEURONS_ALT,
            activation=self.config.ACTIVATION_ALT,
            bias=self.certificate.bias[1],
            config=self.config,
                            )

        lyap_learner._type = CertificateType.RWS.name
        barr_learner._type = CertificateType.BARRIER.name

        return lyap_learner, barr_learner

    def _initialise_optimizer(self):
        
        optimizer = torch.optim.AdamW(
                chain(
                    *(l.parameters() for l in self.learner),
                    ),
                lr=self.config.LEARNING_RATE,
                )
        return optimizer
    
    def _initialise_verifier(self):
        lyap_num_params = sum(p.numel() for p in self.learner[0].parameters() if p.requires_grad)
        barr_num_params = sum(p.numel() for p in self.learner[1].parameters() if p.requires_grad)
        num_params = lyap_num_params + barr_num_params

        verifier_type = verifier.get_verifier_type(self.config.VERIFIER)
        verifier_instance = verifier_type(
                    self.config.N_VARS,
                    self.config.BETA,
                    self.config.N_DATA,
                    num_params,
                    self.config.VERBOSE,
                            )
        return verifier_instance

    def solve(self) -> Result:
        converge_tol = 1e-4
        Sdot = self.S["derivs"]
        S = self.S["states"]
        S_inds = self.S["indices"]
        S_traj = self.S_traj
        times = self.S["times"]
        state = self.init_state(Sdot, S, S_traj, S_inds, times)

        # Reset timers for components
        self.lyap_learner.get_timer().reset()
        
        state["net_dot"] = self.lyap_learner.nn_dot
        iters = 0
        stop = False
        N_data = self.config.N_DATA
        n_test_data = self.config.N_TEST_DATA
        old_loss = float("Inf")
        old_best = float("Inf")
        if self.config.CONVEX_NET:
            state["supps"] = {"active":0,"relaxed":0}
        else:
            state["supps"] = set()
        state["supp_len"] = self.a_priori_supps
        while not stop:
            opt_state_dict = state[ScenAppStateKeys.optimizer].state_dict()
            opt_state_dict["param_groups"][0]["lr"] = 1/(iters+1)
            state[ScenAppStateKeys.optimizer].load_state_dict(opt_state_dict)
            # Legtner component
            
            scenapp_log.debug("\033[1m Lyap Learner \033[0m")
            outputs = self.lyap_learner.get(**state)
            state = {**state, **outputs}
            
            #scenapp_log.debug("\033[1m Barr Learner \033[0m")
            #outputs = self.barr_learner.get(**state) # Alec doesn't  call barr learner for some reason?
            #state = {**state, **outputs}
            
            if self.config.CONVEX_NET:
                state["supps"] = outputs["new_supps"]
            else:
                state["supps"] = state["supps"].union(outputs["new_supps"])
            state = self.update_controller(state)

            # Translator component
            if self.config.CONVEX_NET and torch.abs(state["loss"]-old_loss) < converge_tol:
                scenapp_log.debug("\033[1m Verifier \033[0m")
                

                outputs = self.verifier.get(**state)
                state = {**state, **outputs}

                # Consolidator component # Don't think this is needed/possible for us
                #scenapp_log.debug("\033[1m Consolidator \033[0m")
                #outputs = self.consolidator.get(**state)
                #state = {**state, **outputs}
                print("Epsilon: {:.5f}".format(state[ScenAppStateKeys.bounds]))
                stop = self.process_certificate(S, state, iters)

            elif not self.config.CONVEX_NET and state["best_loss"] == 0.0:
                scenapp_log.debug("\033[1m Verifier \033[0m")
                

                outputs = self.verifier.get(**state)
                state = {**state, **outputs}

                # Consolidator component # Don't think this is needed/possible for us
                #scenapp_log.debug("\033[1m Consolidator \033[0m")
                #outputs = self.consolidator.get(**state)
                #state = {**state, **outputs}
                print("Epsilon: {:.5f}".format(state[ScenAppStateKeys.bounds]))
                stop = self.process_certificate(S, state, iters)

            elif state[ScenAppStateKeys.verification_timed_out]:
                scenapp_log.warning("Verification timed out")
                stop = True
                state[ScenAppStateKeys.bounds] = None
            elif (
                    self.config.SCENAPP_MAX_ITERS <= iters
                    ):
                scenapp_log.warning("Out of iterations")
                stop = True
                state[ScenAppStateKeys.bounds] = None
            elif not self.config.CONVEX_NET and torch.abs(state["best_loss"]-old_best) < converge_tol:
                scenapp_log.info("Convergence reached, but failed to find valid certificate, discarding samples")
                self.discard(state)
                scenapp_log.debug("Discarded {} samples so far".format(len(state["discarded"])))
                iters += 1
                old_loss = state["loss"]
                old_best = state["best_loss"]
                scenapp_log.info("Iteration: {}".format(iters))

            elif not (
                    state[ScenAppStateKeys.found]
                    or state[ScenAppStateKeys.verification_timed_out]
                    ):
                #state = self.process_cex(S, state)

                iters += 1
                old_loss = state["loss"]
                old_best = state["best_loss"]
                scenapp_log.info("Iteration: {}".format(iters))

        state = self.process_timers(state)

        #N_data = sum([S_i.shape[0] for S_i in state[ScenAppStateKeys.S].values()])
        stats = Stats(
                iters, N_data, state["components_times"], torch.initial_seed()
                )
        pre_post = perf_counter()
        a_post_eps = self.a_post_verify(state[ScenAppStateKeys.best_net], state[ScenAppStateKeys.best_net].nn_dot, n_test_data)
        post_time = perf_counter()-pre_post
        print("Direct risk calculation time: {:.5f}s".format(post_time))
        self._result = Result(state[ScenAppStateKeys.bounds], a_post_eps, state[ScenAppStateKeys.best_net], stats)
                #state[ScenAppStateKeys.net], state[ScenAppStateKeys.net_dot], n_test_data)
        return self._result




class ScenApp:
    def __new__(cls, config: ScenAppConfig) -> Union[DoubleScenApp, SingleScenApp]:
        if config.CERTIFICATE in (
                certificate.CertificateType.STABLESAFE,
                certificate.CertificateType.RAR,
                ):
            return DoubleScenApp(config)
        else:
            return SingleScenApp(config)

    
    def __init__(self, config: ScenAppConfig):
        pass

    def solve(self) -> Result:
        raise NotImplementedError("This should be implemented by child classes")
