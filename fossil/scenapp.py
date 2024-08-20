from typing import NamedTuple, Union

import fossil.learner as learner
import fossil.verifier as verifier
from fossil.consts import * 
import fossil.consolidator as consolidator
import fossil.logger as logger
import fossil.certificate as certificate

import torch

import sympy as sp

from scipy.stats import beta as betaF

scenapp_log = logger.Logger.setup_logger(__name__)

class Stats(NamedTuple):
    iters: int
    N_data: int
    times: dict
    seed: int

class Result(NamedTuple):
    res: float 
    cert: learner.LearnerNN
    stats: Stats


class SingleScenApp:
    def __init__(self, config: ScenAppConfig):
        self.config = config
        
        self.x, self.x_map, self.domains = self._initialise_domains()
        self.S, self.S_traj = self._initialise_data() # Needs editing
        self.certificate = self._initialise_certificate()
        self.learner = self._initialise_learner()
        if config.CONVEX_NET:
            self.a_priori_supps = sum([param.numel() for param in self.learner.parameters()]) # Take this and add any violations for convex
        else:
            self.a_priori_supps = None
        self.verifier = self._initialise_verifier() # Need to write my own verifier
        #self.verifier = self._scenapp_verifier
        self.optimizer = self._initialise_optimizer() # Currently not working
        #self.consolidator = self._initialise_consolidator() # Not sure what this 
        #self.translator_type, self.translator = self._initialise_translator()
        #self._result = None
        #self._assert_state()
        if self.config.VERBOSE:
            logger.Logger.set_logger_level(self.config.VERBOSE)
    
    def _initialise_domains(self):
        x = verifier.get_verifier_type(self.config.VERIFIER).new_vars(
            self.config.N_VARS
            )
        #x = [sp.symbols("x"+str(i)) for i in range(self.config.N_VARS)]
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
                    self.certificate.get_supports,
                    self.config.BETA,
                    self.config.N_DATA,
                    self.config.MARGIN,
                    num_params,
                    self.config.VERBOSE,
                            )
        return verifier_instance


    def _initialise_data(self):
        traj_data = self.config.DATA["full_data"]
        state_data = self.config.DATA["states_only"]
        #traj_data = {key: [torch.tensor(elem.T, dtype=torch.float32 ) for elem in self.config.DATA[key]] for key in self.config.DATA} 
        lumped_data = {key: torch.tensor(np.hstack(self.config.DATA["full_data"][key]), dtype=torch.float32 ) for key in self.config.DATA["full_data"]} 
        
        traj_inds = [] 
        curr_ind = 0
        for elem in self.config.DATA["full_data"]["times"]:
            traj_inds.append((curr_ind, curr_ind+len(elem)))
            curr_ind += len(elem)

        #domained_data = {key: [] for key in self.config.DOMAINS}
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
                    
                #domained_data[key] = [elem for elem in lumped_data["states"] if domain.check_containment(elem)]
            if len(domained_data["states"][key]) > 0:
                domained_data["states"][key] = torch.cat((torch.stack(domained_data["states"][key]), state_data[key]))
                domained_data["derivs"][key] = torch.stack(domained_data["derivs"][key])
                domained_data["times"][key] = torch.stack(domained_data["times"][key])
            else:
                domained_data["states"][key] = state_data[key]
        scenapp_log.debug("Data: {}".format(self.config.DATA))
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
        test_data = self.config.DOMAINS["init"]._generate_data(n_data)()

        all_test_data = self.config.SYSTEM().generate_trajs(test_data)
        data = {"states_only": None, "full_data": {"times":all_test_data[0],"states":all_test_data[1],"derivs":all_test_data[2]}}
        
        num_violations = self.certificate.get_supports(cert, cert_deriv, data["full_data"]["states"], data["full_data"]["derivs"], self.config.MARGIN, -1)
        k = num_violations
        beta_bar = (1e-5)/n_data
        N = n_data
        d = 1
        eps = betaF.ppf(1-beta_bar, k+d, N-(d+k)+1) 
        print("A posteriori scenario approach risk: {:.5f}".format(eps))
        print("Violation rate: {:.3f}".format(num_violations/n_data))
        


    def solve(self) -> Result:
        converge_tol = 1e-4
        Sdot = self.S["derivs"]
        S = self.S["states"]
        S_inds = self.S["indices"]
        S_traj = self.S_traj
        # Initialize CEGIS state
        state = self.init_state(Sdot, S, S_traj, S_inds)

        # Reset timers for components
        self.learner.get_timer().reset()
        #self.translator.get_timer().reset()
        #self.verifier.get_timer().reset()
        #self.consolidator.get_timer().reset()
        state["net_dot"] = self.learner.nn_dot
        iters = 0
        stop = False
        N_data = self.config.N_DATA
        n_test_data = self.config.N_TEST_DATA
        old_loss = float("Inf") 
        state["supps"] = set()
        state["supp_len"] = self.a_priori_supps
        while not stop:
            
            opt_state_dict = state[ScenAppStateKeys.optimizer].state_dict()
            opt_state_dict["param_groups"][0]["lr"] = 1/(iters+1)
            state[ScenAppStateKeys.optimizer].load_state_dict(opt_state_dict)
            # Legtner component
            scenapp_log.debug("\033[1m Learner \033[0m")
            outputs = self.learner.get(**state)
            state = {**state, **outputs}
            
            if not self.config.CONVEX_NET:
                state["supps"] = state["supps"].union(outputs["new_supps"])
                #state["supp_len"] = len(state["supps"])
            #print("len supps: {}".format(state["supp_len"]))
            # Update xdot with new controller if necessary
            state = self.update_controller(state)

            # Translator component
            #scenapp_log.debug("\033[1m Translator \033[0m")
            #outputs = self.translator.get(**state)
            #state = {**state, **outputs}

            # Verifier component
            #print(state["loss"]) # Finding loss = 0, not certain why... Maybe just learning a flat lyapunov?

            #if state[ScenAppStateKeys.bounds] <= self.config.EPS: # Check for convergence in loss instead...
            #    stop = self.process_certificate(S, state, iters)
            if torch.abs(state["loss"]-old_loss) < converge_tol:
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
                    self.config.SCENAPP_MAX_ITERS == iters
                    ):
                scenapp_log.warning("Out of iterations")
                stop = True
                state[ScenAppStateKeys.bounds] = None

            elif not (
                    state[ScenAppStateKeys.found]
                    or state[ScenAppStateKeys.verification_timed_out]
                    ):
                #state = self.process_cex(S, state)

                iters += 1
                old_loss = state["loss"]
                scenapp_log.info("Iteration: {}".format(iters))

        state = self.process_timers(state)

        #N_data = sum([S_i.shape[0] for S_i in state[ScenAppStateKeys.S].values()])
        stats = Stats(
                iters, N_data, state["components_times"], torch.initial_seed()
                )
        self._result = Result(state[ScenAppStateKeys.bounds], state["net"], stats)
        self.a_post_verify(state[ScenAppStateKeys.net], state[ScenAppStateKeys.net_dot], n_test_data)
        return self._result

    def init_state(self, Sdot, S, S_traj, S_inds):
        state = {
                ScenAppStateKeys.net: self.learner,
                ScenAppStateKeys.optimizer: self.optimizer,
                ScenAppStateKeys.S: S,
                ScenAppStateKeys.S_dot: Sdot,
                ScenAppStateKeys.S_traj: S_traj["states"],
                ScenAppStateKeys.S_traj_dot: S_traj["derivs"],
                ScenAppStateKeys.S_inds: S_inds,
                ScenAppStateKeys.V: None,
                ScenAppStateKeys.V_dot: None,
                ScenAppStateKeys.x_v_map: self.x_map,
                #ScenAppStateKeys.xdot: self.xdot,
                #ScenAppStateKeys.xdot_func: self.f._f_torch,
                ScenAppStateKeys.found: False,
                ScenAppStateKeys.verification_timed_out: False,
                ScenAppStateKeys.trajectory: None,
                ScenAppStateKeys.ENet: self.config.ENET,
                ScenAppStateKeys.margin: self.config.MARGIN
                }

        return state

    def process_timers(self, state: dict[str, Any]) -> dict[str, Any]:
        state[ScenAppStateKeys.components_times] = [
                self.learner.get_timer().sum,
                #self.translator.get_timer().sum,
                self.verifier.get_timer().sum,
                #self.consolidator.get_timer().sum,
                ]
        scenapp_log.info("Learner times: {}".format(self.learner.get_timer()))
        #cegis_log.info("Translator times: {}".format(self.translator.get_timer()))
        scenapp_log.info("Verifier times: {}".format(self.verifier.get_timer()))
        #cegis_log.info("Consolidator times: {}".format(self.consolidator.get_timer()))
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

        if self.config.CERTIFICATE == CertificateType.RSWS:
            stay = self.certificate.beta_search(
                    self.learner,
                    self.verifier,
                    state[ScenAppStateKeys.V],
                    state[ScenAppStateKeys.V_dot],
                    S,
                    )
            # Only stop if we prove the final stay condition
            stop = stay
            if stay:
                print(f"Found a valid {self.config.CERTIFICATE.name} certificate")
            else:
                print(
                        f"Found a valid RWS certificate, but could not prove the final stay condition. Keep searching..."
                        )
                state[ScenAppStateKeys.found] = False
                if self.config.SCENAPP_MAX_ITERS == iters:
                    stop = True
        else:
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

class DoubleScenApp:
    pass

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
