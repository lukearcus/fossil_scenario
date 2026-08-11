#test
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
import gc
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
        torch.set_num_threads(self.config.N_THREADS)

        self.x, self.x_map, self.domains = self._initialise_domains()
        self.S, self.S_traj, self.init_S = self._initialise_data(self.config.DATA["full_data"], self.config.DATA["states_only"]) # Needs editing
        self.certificate = self._initialise_certificate()
        self.learner = self._initialise_learner()
        self.a_priori_supps = None
        self.verifier = self._initialise_verifier() 
        self.optimizer = self._initialise_optimizer() 
        #self._pretrain_controller()
        if self.config.VERBOSE:
            logger.Logger.set_logger_level(self.config.VERBOSE)
    
    def _pretrain_controller(self):
        num_pretrain_loops = 10000
        all_data = torch.cat([self.S["states"][key] for key in self.S["states"]])
        exp_control = torch.tensor([self.config["SYSTEM"][0].controller(0,datum) for datum in all_data])[:,None] # to mahe 2d, shouldn't be needed really
        for i in range(num_pretrain_loops):
            self.optimizer[1].zero_grad()
            nn_control = self.learner[1](all_data)
            loss = torch.norm(exp_control-nn_control, dim=1).sum()
            loss.backward()
            self.optimizer[1].step()
            if i % 100 == 0:
                print("Pretrain loss on step {}: {}".format(i,loss))



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
         
         V = learner.DissV(
            self.config.N_VARS,
            self.certificate.learn,
            self.config.N_HIDDEN_NEURONS["V"],
            activation=self.config.ACTIVATION["V"],
            bias=self.certificate.bias,
            config=self.config,
            )
         if self.config.CERTIFICATE == certificate.CertificateType.DISSIPATIVITY:
            Q = learner.DissQ(
               self.config.N_VARS,
               self.certificate.learn,
               self.config.N_HIDDEN_NEURONS["Q"],
               activation=self.config.ACTIVATION["Q"],
               bias=self.certificate.bias,
               config=self.config,
                               )
            S = learner.DissS(
               self.config.N_VARS,
               self.config.CONTROL_VARS,
               self.certificate.learn,
               self.config.N_HIDDEN_NEURONS["S"],
               activation=self.config.ACTIVATION["S"],
               bias=self.certificate.bias,
               config=self.config,
                               )
            R = learner.DissR(
               self.config.N_VARS,
               self.config.CONTROL_VARS,
               self.certificate.learn,
               self.config.N_HIDDEN_NEURONS["R"],
               activation=self.config.ACTIVATION["R"],
               bias=self.certificate.bias,
               config=self.config,
                               )
            L = learner.DissS(
               self.config.N_VARS,
               self.config.N_VARS,
               self.certificate.learn,
               self.config.N_HIDDEN_NEURONS["L"],
               activation=self.config.ACTIVATION["L"],
               bias=self.certificate.bias,
               config=self.config,
                               )
            return (V, Q, S, R, L)
         elif self.config.CERTIFICATE == certificate.CertificateType.DIRECTCONTROL:
            u = learner.Controller(
               self.config.N_VARS,
               self.config.CONTROL_VARS,
               self.certificate.learn,
               self.config.N_HIDDEN_NEURONS["u"],
               activation=self.config.ACTIVATION["u"],
               bias=self.certificate.bias,
               config=self.config,
                               )
            return (V, u)
         elif self.config.CERTIFICATE == certificate.CertificateType.DIRECTCONTROLBARR:
            u = learner.Controller(
               self.config.N_VARS,
               self.config.CONTROL_VARS,
               self.certificate.learn,
               self.config.N_HIDDEN_NEURONS["u"],
               activation=self.config.ACTIVATION["u"],
               bias=self.certificate.bias,
               config=self.config,
                               )
            return (V, u)
         elif self.config.CERTIFICATE == certificate.CertificateType.DIRECTCONTROLRWA:
            u = learner.Controller(
               self.config.N_VARS,
               self.config.CONTROL_VARS,
               self.certificate.learn,
               self.config.N_HIDDEN_NEURONS["u"],
               activation=self.config.ACTIVATION["u"],
               bias=self.certificate.bias,
               config=self.config,
                               )
            return (V, u)
         else:
            raise NotImplementedError

    def _initialise_verifier(self):
        num_params = sum(sum(p.numel() for p in l.parameters() if p.requires_grad) for l in self.learner)

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
        lumped_data = {key: torch.tensor(np.concatenate(traj_data[key], axis=-1), dtype=torch.float32 ) for key in traj_data} 
        #lumped_data["g_vals"] = torch.tensor(np.stack(traj_data["g_vals"]), dtype=torch.float32)
        inits = np.stack(traj_data["states"])[:,:,0]
        traj_inds = []
        curr_ind = 0
        for elem in traj_data["times"]:
            if type(elem) is not np.float64:
                elem_len = len(elem)
            else:
                elem_len = 1
            traj_inds.append((curr_ind, curr_ind+elem_len))
            curr_ind += elem_len

        # Precompute traj start boundaries for O(log n) traj lookup via searchsorted
        # (replaces the per-sample linear scan `if ind in range(*index)` over trajectories).
        traj_starts = np.array([s for s, _ in traj_inds], dtype=np.int64)
        n_trajs = len(traj_inds)

        domained_data = {"states":{},"times":{},"derivs":{}, "indices":{}, "f":{}, "g":{}}
        max_len = len(traj_data["times"][0])
        states_all = lumped_data["states"]            # (n_dims, total_samples)
        total_samples = states_all.shape[1]
        for key in self.config.DOMAINS:
            domain = self.config.DOMAINS[key]
            # Initialise sub-entries to empty lists to match original semantics when a domain
            # has no contained trajectory points (downstream code does len(Sdot[key]) etc.).
            domained_data["derivs"][key] = []
            domained_data["times"][key] = []
            domained_data["f"][key] = []
            domained_data["g"][key] = []
            # Vectorised containment check over the whole batch at once (replaces the
            # per-sample Python loop with one domain.check_containment call).
            contained = domain.check_containment(states_all.T)  # (total_samples,) bool
            contained = contained.bool() if hasattr(contained, 'bool') else torch.as_tensor(contained).bool()
            contained_inds = torch.where(contained)[0].numpy()  # global sample indices, sorted
            domained_data["indices"][key] = [torch.ones(max_len, dtype=torch.int32) * (-1) for _ in traj_inds]

            if contained_inds.shape[0] > 0:
                # Map each contained sample to its trajectory via searchsorted on starts.
                traj_of = np.searchsorted(traj_starts, contained_inds, side='right') - 1
                # Running counter per traj (local position within each traj's contained samples).
                traj_counts = np.zeros(n_trajs, dtype=np.int64)
                for p, ind in enumerate(contained_inds):
                    t = traj_of[p]
                    domained_data["indices"][key][t][traj_counts[t]] = p
                    traj_counts[t] += 1
                # Gather the contained slices in order.
                domained_data["states"][key] = states_all[:, contained_inds].T
                domained_data["derivs"][key] = lumped_data["derivs"][:, contained_inds].T
                domained_data["times"][key] = lumped_data["times"][contained_inds]
                domained_data["f"][key] = lumped_data["f_vals"][:, contained_inds].T
                domained_data["g"][key] = lumped_data["g_vals"][:, :, contained_inds].permute(2, 0, 1)
                if key in state_data:
                    domained_data["states"][key] = torch.cat((domained_data["states"][key], state_data[key]))
            else:
                domained_data["states"][key] = state_data[key]
            # Replace remaining -1 sentinels in each traj's index array with index[0]
            # (matches the original: `index[index==-1] = index[0]`).
            new_inds = []
            for index in domained_data["indices"][key]:
                index[index==-1] = index[0]
                new_inds.append(index)
            domained_data["indices"][key] = new_inds

        return domained_data, traj_data, inits


    def _initialise_optimizer(self):
        #return torch.optim.SGD(
        optimizers = []
        for i, l in enumerate(self.learner):
            optimizers.append(torch.optim.AdamW(
                [{"params": l.parameters()}], # Might need to change this to consider controller parameters
                lr=self.config.LEARNING_RATE[i],
                ))
        return optimizers
        #return (torch.optim.AdamW(
        #        [{"params": l.parameters()}], # Might need to change this to consider controller parameters
        #        lr=self.config.LEARNING_RATE,
        #        ) for l in self.learner)


    def a_post_verify(self, certs, n_data):
        state_data = self.config.DATA["states_only"]
        torch.manual_seed(clock_gettime(0))      #allows different samples when running in parallel
        try:
            test_data = self.config.DOMAINS["init"]._generate_data(n_data)()
        except KeyError:
            test_data = self.config.DOMAINS["lie"]._generate_data(n_data)()
        
        def control(t, x):
            x = torch.tensor(x,dtype=torch.float32)
            if len(x.shape) == 1:
                return certs[1](x.unsqueeze(1).T).detach().numpy()
            else:
                return certs[1](x.unsqueeze(2).mT).detach().numpy()

        new_systems = [self.config.SYSTEM[0].__new__(self.config.SYSTEM[0].__class__) for i in test_data]
        for sys in new_systems:
            sys.__init__()
            sys.controller = control
        all_test_data = [sys.generate_trajs(np.expand_dims(test_datum,0)) for sys, test_datum in zip(new_systems, test_data)]
        
        #all_data = [system.generate_trajs(np.expand_dims(init_datum,0)) for system, init_datum in zip(self.config.SYSTEM, self.init_S)]
    
        times =  [datum[0][0] for datum in all_test_data]
        states = [datum[1][0] for datum in all_test_data]
        derivs = [datum[2][0] for datum in all_test_data]
        f_vals = [datum[3][0] for datum in all_test_data]
        g_vals = [datum[4][0] for datum in all_test_data]
        
        #new_traj_data = {"times":times,"states":states,"derivs":derivs, "f_vals":f_vals, "g_vals":g_vals}        
       # data = {"states_only": None, "full_data": {"times":all_test_data[0],"states":all_test_data[1],"derivs":all_test_data[2]}}
        data = {"states_only": None, "full_data": {"times":times,"states":states,"derivs":derivs, "f_vals":f_vals, "g_vals":g_vals}}
        #data = {"states_only": None, "full_data": {"times":all_test_data[0],"states":all_test_data[1],"derivs":all_test_data[2], "f_vals":all_test_data[3], "g_vals":all_test_data[4]}}
        num_violations, true_violations = self.certificate.get_violations(certs, data["full_data"], state_data)
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

    def update_controller(self, state):
        
        if self.config.CERTIFICATE == certificate.CertificateType.DISSIPATIVITY:
        
            def diss_control(t, x):
                x = torch.tensor(x,dtype=torch.float32)
                if len(x.shape) == 1:
                    R = state["best_net"][3](x.unsqueeze(1).T).detach()
                    return (-torch.inverse(R)@state["best_net"][2](x.unsqueeze(1).T)).detach().numpy()
                else:
                    R = state["best_net"][3](x.unsqueeze(2).mT).detach()
                    return (-torch.bmm(torch.inverse(R),state["best_net"][2](x.unsqueeze(2).mT))).detach().numpy()

            for sys in self.config.SYSTEM:
                sys.controller = diss_control
        elif (self.config.CERTIFICATE == certificate.CertificateType.DIRECTCONTROL or self.config.CERTIFICATE == certificate.CertificateType.DIRECTCONTROLBARR) or self.config.CERTIFICATE == certificate.CertificateType.DIRECTCONTROLRWA:

            def control(t, x):
                x = torch.tensor(x,dtype=torch.float32)
                
                #controls = torch.arange(self.u_min,self.u_max,0.01).
                #nexts_spaced = (self.f.mT+torch.bmm(self.g.mT, torch.arange(self.u_min,self.u_max,0.01).unsqueeze(0).repeat(1,1,1))).mT
                #V_next_min_space_ind = state["best_net"][0](nexts_spaced.flatten(0,1)).reshape(g_samples.shape[0],-1).min(axis=1)[1]
                #return controls[V_next_min_space_ind]
                # NOTE: previously referenced V_next_min_space here, which was never defined
                # (UnboundLocalError). The commented-out space-search above would have computed
                # it; the direct-NN return below doesn't need it.

                if len(x.shape) == 1:
                    return state["best_net"][1](x.unsqueeze(1).T).detach().numpy()
                else:
                    return state["best_net"][1](x.unsqueeze(2).mT).detach().numpy()

            for sys in self.config.SYSTEM:
                sys.controller = control
        else:
            return state
        all_data = [system.generate_trajs(np.expand_dims(init_datum,0)) for system, init_datum in zip(self.config.SYSTEM, self.init_S)]
    
        times =  [datum[0][0] for datum in all_data]
        states = [datum[1][0] for datum in all_data]
        derivs = [datum[2][0] for datum in all_data]
        f_vals = [datum[3][0] for datum in all_data]
        g_vals = [datum[4][0] for datum in all_data]
        
        new_traj_data = {"times":times,"states":states,"derivs":derivs, "f_vals":f_vals, "g_vals":g_vals}        
        #import pdb; pdb.set_trace()
        #if state["learners"][0](new_traj_data["states"]).min() < state["learners"][0](self.S_traj["states"]).min(): 
        self.S, self.S_traj, _ = self._initialise_data(new_traj_data, self.config.DATA["states_only"]) # Needs editing
        
        state[ScenAppStateKeys.S] = self.S["states"]
        state[ScenAppStateKeys.S_dot] = self.S["derivs"]
        state[ScenAppStateKeys.S_traj] = self.S_traj["states"]
        state[ScenAppStateKeys.S_traj_dot] =  self.S_traj["derivs"]
        state[ScenAppStateKeys.S_inds] =  self.S["indices"]
        state[ScenAppStateKeys.times] = self.S["times"]
        state[ScenAppStateKeys.f] = self.S["f"]
        state[ScenAppStateKeys.g] = self.S["g"]

        if self.config.CERTIFICATE == certificate.CertificateType.DIRECTCONTROL:
            acc = sum([any(self.config.DOMAINS[DomainNames.XG.value].check_containment(torch.tensor(traj.T))) for traj in self.S_traj["states"]])/ len(self.S_traj["states"]) * 100
        elif self.config.CERTIFICATE == certificate.CertificateType.DIRECTCONTROLBARR:
            acc = (1-sum([any(self.config.DOMAINS[DomainNames.XU.value].check_containment(torch.tensor(traj.T))) for traj in self.S_traj["states"]])/ len(self.S_traj["states"])) * 100
        elif self.config.CERTIFICATE == certificate.CertificateType.DIRECTCONTROLRWA:
            acc = sum([any(self.config.DOMAINS[DomainNames.XG.value].check_containment(torch.tensor(traj.T))) and not any(self.config.DOMAINS[DomainNames.XU.value].check_containment(torch.tensor(traj.T))) for traj in self.S_traj["states"]])/ len(self.S_traj["states"]) * 100
            #acc = (1-sum([any(self.config.DOMAINS[DomainNames.XU.value].check_containment(torch.tensor(traj.T))) for traj in self.S_traj["states"]])/ len(self.S_traj["states"])) * 100
        scenapp_log.info("Controller accuracy: {:.5f}%".format(acc))
        scenapp_log.info(self.S_traj["states"][-1].T)
        
        print(self.S_traj["states"][-1].T)
        #if acc > 99:
        #    import pdb; pdb.set_trace()
        return state

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
                    return state
                self.remaining_inds=list(set(self.remaining_inds)-to_remove)
            state["supps"] = set()
        new_traj_inds = self.remaining_inds
        new_traj_data = {}
        for key in traj_data:
            new_traj_data[key] = [traj_data[key][ind] for ind in new_traj_inds]
            self.S, self.S_traj, self.init_S = self._initialise_data(new_traj_data, self.config.DATA["states_only"]) # Needs editing
        
        state[ScenAppStateKeys.S] = self.S["states"]
        state[ScenAppStateKeys.S_dot] = self.S["derivs"]
        state[ScenAppStateKeys.S_traj] = self.S_traj["states"]
        state[ScenAppStateKeys.S_traj_dot] =  self.S_traj["derivs"]
        state[ScenAppStateKeys.S_inds] =  self.S["indices"]
        state[ScenAppStateKeys.times] = self.S["times"]
        state[ScenAppStateKeys.f] = self.S["f"]
        state[ScenAppStateKeys.g] = self.S["g"]
        return state

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
        f = self.S["f"]
        g = self.S["g"]
        # Initialize CEGIS state
        state = self.init_state(Sdot, S, S_traj, S_inds, times, f, g)
        # Track the (flattened, detached) parameter vector of the best net so we can detect when
        # the network has actually stopped changing between outer iterations. Once the net has
        # converged and best_loss <= margin, an additional learn loop has effectively confirmed
        # that controller + certificate together give zero loss, and we can verify.
        param_vec = torch.cat([p.detach().flatten() for l in state["best_net"] for p in l.parameters()])

        # Reset timers for components
        for l in self.learner:
            l.get_timer().reset()
        
        state["net_dot"] = self.learner[0].nn_dot
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
        start_switch = False
        margin = self.config.MARGIN
        j = 0
        old_nets = copy.deepcopy(self.learner)
        reverted=False
        # NOTE: do NOT call update_controller here — it would replace the initial (reference)
        # trajectories with random-controller trajectories before the CEGIS loop starts.
        start_time = perf_counter()
        #for param in self.learner[1].parameters():
        #    param.requires_grad=False
        while not stop:
            if perf_counter() - start_time > self.config.SCENAPP_MAX_TIME_S:
                scenapp_log.warning("Out of time (SCENAPP_MAX_TIME_S={})".format(self.config.SCENAPP_MAX_TIME_S))
                stop = True
                state[ScenAppStateKeys.bounds] = None
                break
            scenapp_log.debug("\033[1m Learner \033[0m")
            # Maybe switching is a good idea???

            #if start_switch:
            #    if j == 0:
            #        state["parallel"] = False
            #    elif j == 10:
            #        state["parallel"] = True
            #        j= 0
            #        start_switch = False
            #    j += 1
            
            #if iters % 2:
            #    for param in self.learner[0].parameters():
            #        param.requires_grad=False
            #    for param in self.learner[1].parameters():
            #        param.requires_grad=True
            #else:
            #    for param in self.learner[1].parameters():
            #        param.requires_grad=False
            #    for param in self.learner[0].parameters():
            #        param.requires_grad=True
            beta = self.learner[0](self.S["states"]["goal_border"]).max()
            # On iteration 0, self.S_traj holds the REFERENCE trajectories (generated by the
            # reference controller in the experiment script), not the learned NN controller.
            # Do not freeze u1 based on reference trajectories — force controller_training=True
            # so u1 is actually trained, and only freeze once the LEARNED controller's
            # regenerated trajectories reach XG (iters > 0).
            if iters > 0 and all([any(self.config.DOMAINS[DomainNames.XG.value].check_containment(torch.tensor(traj.T))) for traj in self.S_traj["states"]]):
            
                for param in self.learner[1].parameters():
                    param.requires_grad=False
                scenapp_log.info("Controller update off")
                controller_training = False
            else:
                for param in self.learner[1].parameters():
                    param.requires_grad=True
                scenapp_log.info("Controller update on")
                controller_training = True

            outputs = self.learner[0].get(**state)
            state = {**state, **outputs}
            #if old_best < state["best_loss"] and state["best_loss"]>margin:
            if False:
                print("Increased loss, reverting")
                state["best_loss"] = old_best
                self.learner = copy.deepcopy(old_nets)
                state["best_net"] = copy.deepcopy(old_nets)
                reverted=True # check this is needed??? Seems to work without??..
                #Also had old_best updating on reversion before when success?
            else:
                old_best = state["best_loss"]
                old_nets = copy.deepcopy(state["best_net"])
                reverted=False

            if state["best_loss"] >margin:
                scenapp_log.info("Best loss: {:.10f}".format(state["best_loss"]))
            else:
                scenapp_log.info("Best Loss below margin")
                #if state["parallel"]==True:
                #    start_switch = True
            if isinstance(old_best, (int, float)):
                scenapp_log.info("Previous Best loss: {:.10f}".format(old_best))
            else:
                scenapp_log.info("Previous Best loss: {:.10f}".format(old_best.item()))
            # param_delta is logged for diagnostics only. The verification gate is now:
            #   best_loss <= margin (V Lyapunov condition holds on the support subset) AND
            #   not controller_training (trajectories geometrically reach XG).
            # The latter is the goal-reaching check; the former is the scenario-approach
            # certificate condition. Together: "Epsilon printed + goal reached".
            new_param_vec = torch.cat([p.detach().flatten() for l in state["best_net"] for p in l.parameters()])
            param_delta = (new_param_vec - param_vec).norm().item()
            scenapp_log.debug("Param delta (rel): {:.6e} / {:.6e}".format(param_delta, self.config.CONVERGE_TOL * (param_vec.norm().item() + 1e-12)))
            param_vec = new_param_vec

            if state["best_loss"] <= margin and controller_training:
                # Regenerate trajectories with the improved controller and continue training.
                # Once trajectories reach XG, controller_training flips False and the gate below
                # can fire.
                scenapp_log.info("Updating controller")
                state = self.update_controller(state)

            state["supps"] = state["supps"].union(outputs["new_supps"])

            if state["best_loss"] <= margin and not controller_training:
            #if True:
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
            #elif torch.abs(old_best-state["best_loss"]) < converge_tol:
            elif state["best_loss"] > margin and old_best-state["best_loss"] < converge_tol:
                scenapp_log.info("Convergence reached, but failed to find valid certificate, discarding samples")
                #state = self.discard(state)
                #scenapp_log.debug("Discarded {} samples so far".format(len(state["discarded"])))
                iters += 1
                old_loss = state["loss"]
                old_best = state["best_loss"]
                scenapp_log.info("Iteration: {}".format(iters))
                for (net, best) in zip(state[ScenAppStateKeys.net], state["best_net"]):
                    net.load_state_dict(best.state_dict())
                #= copy.deepcopy(state["best_net"])

            elif not (
                    state[ScenAppStateKeys.found]
                    or state[ScenAppStateKeys.verification_timed_out]
                    ):

                iters += 1
                old_loss = state["loss"]
                old_best = state["best_loss"]
                scenapp_log.info("Iteration: {}".format(iters))
            if state["loss"].item() == 0:
                scenapp_log.info("Zero Current Loss")
            else:
                scenapp_log.info("Current loss: {:.10f}".format(state["loss"].item()))
            gc.collect()
        state = self.process_timers(state)

        stats = Stats(
                iters, N_data, state["components_times"], torch.initial_seed()
                )
        pre_post = perf_counter()
        a_post_eps = self.a_post_verify(state[ScenAppStateKeys.best_net], n_test_data)
        print("Direct property guarantee time: {:.5f}s".format(perf_counter()-pre_post))
        self._result = Result(state[ScenAppStateKeys.bounds], a_post_eps, state[ScenAppStateKeys.best_net], stats)
                #state[ScenAppStateKeys.net], state[ScenAppStateKeys.net_dot], n_test_data)
        return self._result

    def init_state(self, Sdot, S, S_traj, S_inds, times, f, g):
        state = {
                ScenAppStateKeys.net: self.learner,
                ScenAppStateKeys.optimizer: self.optimizer,
                ScenAppStateKeys.S: S,
                ScenAppStateKeys.S_dot: Sdot,
                ScenAppStateKeys.S_traj: S_traj["states"],
                ScenAppStateKeys.S_traj_dot: S_traj["derivs"],
                ScenAppStateKeys.S_inds: S_inds,
                ScenAppStateKeys.f: f, 
                ScenAppStateKeys.g: g , 
                ScenAppStateKeys.times: times,
                ScenAppStateKeys.V: None,
                ScenAppStateKeys.V_dot: None,
                ScenAppStateKeys.x_v_map: self.x_map,
                ScenAppStateKeys.found: False,
                ScenAppStateKeys.verification_timed_out: False,
                ScenAppStateKeys.trajectory: None,
                ScenAppStateKeys.ENet: self.config.ENET,
                ScenAppStateKeys.best_loss: np.inf,
                ScenAppStateKeys.best_net: self.learner,
                ScenAppStateKeys.discarded: set(),
                ScenAppStateKeys.convex: self.config.CONVEX_NET,
                ScenAppStateKeys.discrete: self.config.TIME_DOMAIN != TimeDomain.CONTINUOUS,
                ScenAppStateKeys.parallel: self.config.PARALLEL
                }

        return state

    def process_timers(self, state: dict[str, Any]) -> dict[str, Any]:
        state[ScenAppStateKeys.components_times] = [
                self.learner[0].get_timer().sum,
                self.verifier.get_timer().sum,
                ]
        print("Learner times: {}".format(self.learner[0].get_timer()))
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

            gc.collect()
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
