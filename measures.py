from operator import sub
from numba.cuda import test
import numpy as np
import warnings
import time
import itertools
import multiprocessing as mp

from scipy.optimize import minimize
from scipy.sparse import data
from JointProbabilityMatrix import JointProbabilityMatrix, ConditionalProbabilities
from params_matrix import matrix2params_incremental,params2matrix_incremental
from save_min_path_class import Simulator

def synergistic_entropy_upper_bound(self, variables=None):
    """
    For a derivation, see Quax et al. (2015). No other stochastic variable can have more than this amount of
    synergistic information about the stochastic variables defined by this pdf object, or the given subset of
    variables in <variables>.
    :type variables: list of int
    :rtype: float
    """
    if variables is None:
        variables = range(len(self))

    indiv_entropies = [self.entropy([var]) for var in variables]
    return float(self.entropy() - max(indiv_entropies))

def append_variables_using_state_transitions_table(self, state_transitions):
    """
    Append one or more stochastic variables to this joint pdf, whose conditional pdf is defined by the provided
    'state transitions table'. In the rows of this table the first <self.numvariables> values are the values for
    the pre-existing stochastic variables; the added values are taken to be the deterministically determined
    added variable values, i.e., Pr(appended_vars = X, current_vars) = Pr(current_vars) so that
    Pr(appended_vars = X | current_vars) = 1 for whatever X you appended, where X is a list of values.
    :param state_transitions: list of lists, where each sublist is of
    length self.numvariables + [num. new variables] and is a list of values, each value in [0, numvalues),
    where the first self.numvariables are the existing values ('input') and the remaining are the new variables'
    values ('output').

    Can also provide a function f(values, num_values) which returns a list of values for the to-be-appended
    stochastic variables, where the argument <values> is a list of values for the existing variables (length
    self.numvariables).
    :type state_transitions: list or function
    """

    lists_of_possible_given_values = [range(self.numvalues) for _ in range(self.numvariables)]
    if hasattr(state_transitions, '__call__'):
        state_transitions = [list(existing_vars_values) + list(state_transitions(existing_vars_values,
                                                                                 self.numvalues))
                             for existing_vars_values in itertools.product(*lists_of_possible_given_values)]

    extended_joint_probs = np.zeros([self.numvalues]*len(state_transitions[0]))

    # todo this for debugging? cycle through all possible values for self.numvariables and see if it is present
    # in the state_transitions
    # lists_of_possible_states_per_variable = [range(self.numvalues) for variable in xrange(self.numvariables)]

    # one row should be included for every possible set of values for the pre-existing stochastic variables
    assert len(state_transitions) == np.power(self.numvalues, self.numvariables)

    for states_row in state_transitions:
        assert len(states_row) > self.numvariables, 'if appending then more than self.numvariables values ' \
                                                    'should be specified'
        assert len(states_row) == len(state_transitions[0]), 'not all state rows of equal length; ' \
                                                             'appending how many variables? Make up your mind. '

        # probability that the <self.numvariables> of myself have the values <state_transitions[:self.numvariables]>
        curvars_prob = self(states_row[:self.numvariables])

        assert 0.0 <= curvars_prob <= 1.0, 'probability not in 0-1'

        # set Pr(appended_vars = X | current_vars) = 1 for one set of values for the appended variables (X) and 0
        # otherwise (which is already by default), so I setting here
        # Pr(appended_vars = X, current_vars) = Pr(current_vars)
        extended_joint_probs[tuple(states_row)] = curvars_prob

    assert np.ndim(extended_joint_probs) == len(state_transitions[0])
    if len(state_transitions[0]) > 1:
        assert len(extended_joint_probs[0]) == self.numvalues
        assert len(extended_joint_probs[-1]) == self.numvalues
    
    if __debug__:
        np.testing.assert_almost_equal(np.sum(extended_joint_probs), 1.0)

    self.reset(len(state_transitions[0]), self.numvalues, extended_joint_probs)

def append_random_srv(self,parameter_values_before,num_syn):
    self_with_random_srv = JointProbabilityMatrix(self.numvariables+num_syn,self.numvalues)
    random_pars = np.random.random(self.numvalues**(self.numvariables+num_syn)-(self.numvalues**self.numvariables))
    params2matrix_incremental(self_with_random_srv,parameter_values_before + list(random_pars))
    return self_with_random_srv

def append_synergistic_variables(self,parX,data,num_synergistic_variables,summed_modulo=False,\
                                        subject_variables=None,num_repeats=1,\
                                        agnostic_about=None,minimize_method=None,\
                                        tol_nonsyn_mi_frac=0.05, tol_agn_mi_frac=0.05,\
                                        initial_guesses=[],multi=False):
    """
    Append <num_synergistic_variables> variables in such a way that they are agnostic about any individual
    existing variable (one of self.numvariables thus) but have maximum MI about the set of self.numvariables
    variables taken together.
    :param minimize_method:
    :param tol_nonsyn_mi_frac: set to None for computational speed
    :param tol_agn_mi_frac: set to None for computational speed
    :return:
    :param agnostic_about: a list of variable indices to which the new synergistic variable set should be
    agnostic (zero mutual information). This can be used to find a 'different', second SRV after having found
    already a first one. If already multiple SRVs have been found then you can choose between either being agnostic
    about these previous SRVs jointly (so also possibly any synergy among them), or being 'pairwise' agnostic
    to each individual one, in which case you can pass a list of lists, then I will compute the added cost for
    each sublist and sum it up.
    :param num_repeats:
    :param subject_variables: the list of variables the <num_synergistic_variables> should be synergistic about;
    then I think the remainder variables the <num_synergistic_variables> should be agnostic about. This way I can
    support adding many UMSRVs (maybe make a new function for that) which then already are orthogonal among themselves,
    meaning I do not have to do a separate orthogonalization for the MSRVs as in the paper's theoretical part.
    :param num_synergistic_variables:
    :param initial_guess_summed_modulo:
    :param verbose:
    :return:
    """
    if not agnostic_about is None:
        if len(agnostic_about) == 0:
            agnostic_about = None  # treat as if not supplied

    assert subject_variables is not None, 'define subject variables!'

    pdf_with_srvs = self.copy()
    pdf_with_srvs_for_optimization = pdf_with_srvs.copy()
    pdf_subjects_snew = pdf_with_srvs_for_optimization[subject_variables]
    append_variables_using_state_transitions_table(pdf_subjects_snew,
        state_transitions=lambda vals, mv: [int(np.mod(np.sum(vals), mv))]*num_synergistic_variables)
    num_free_parameters = (self.numvalues**(num_synergistic_variables+len(subject_variables))) - 1 - len(parX)
    upper_bound_synergistic_information = synergistic_entropy_upper_bound(self,subject_variables)
    if not agnostic_about is None:
        # upper_bound_agnostic_information is only used to normalize the cost term for non-zero MI with
        # the agnostic_variables (evidently a SRV is sought that has zero MI with these)
        if np.ndim(agnostic_about) == 1:
            upper_bound_agnostic_information = self.entropy(agnostic_about)
        elif np.ndim(agnostic_about) == 2:
            # in this case the cost term is the sum of MIs of the SRV with the sublists, so max cost term is this..
            upper_bound_agnostic_information = sum([self.entropy(ai) for ai in agnostic_about])
    else:
        upper_bound_agnostic_information = 0  # should not even be used...

    # todo: should lower the upper bounds if the max possible entropy of the SRVs is less...
    assert upper_bound_synergistic_information != 0.0, 'can never find any SRV!'

    # in the new self, these indices will identify the synergistic variables that will be added
    synergistic_variables = range(len(self), len(self) + num_synergistic_variables)
    # print("P3 UPPERBOUND",upper_bound_synergistic_information)
    
    def cost_func_subjects_only(free_params, parameter_values_before, extra_cost_rel_error=True):
        """
        This cost function searches for a Pr(S,Y,A,X) such that X is synergistic about S (subject_variables) only.
        This fitness function also taxes any correlation between X and A (agnostic_variables), but does not care
        about the relation between X and Y.
        :param free_params:
        :param parameter_values_before:
        :return:
        """
        assert len(free_params) == num_free_parameters

        if min(free_params) < -0.00001 or max(free_params) > 1.00001:
            warnings.warn('scipy\'s minimize() is violating the parameter bounds 0...1 I give it: '
                        + str(free_params))

            # high cost for invalid parameter values
            # note: maximum cost normally from this function is about 2.0
            return 10.0 + 100.0 * np.sum([p - 1.0 for p in free_params if p > 1.0]
                                        + [np.abs(p) for p in free_params if p < 0.0])

        # assert max(free_params) <= 1.00001, \
        #     'scipy\'s minimize() is violating the parameter bounds 0...1 I give it: ' + str(free_params)

        free_params = [min(max(fp, 0.0), 1.0) for fp in free_params]  # clip small roundoff errors
        params2matrix_incremental(pdf_subjects_snew,list(parameter_values_before) + list(free_params))

        # this if-block will add cost for the estimated amount of synergy induced by the proposed parameters,
        # and possible also a cost term for the ratio of synergistic versus non-synergistic info as extra
        if not subject_variables is None:
            assert pdf_subjects_snew.numvariables == len(subject_variables) + num_synergistic_variables

            # this can be considered to be in range [0,1] although particularly bad solutions can go >1
            if not extra_cost_rel_error:
                cost = (upper_bound_synergistic_information - pdf_subjects_snew.synergistic_information_naive(
                    variables_SRV=range(len(subject_variables), len(pdf_subjects_snew)),
                    variables_X=range(len(subject_variables)))) / upper_bound_synergistic_information
            else:
                tot_mi = pdf_subjects_snew.mutual_information(range(len(subject_variables), len(pdf_subjects_snew)),
                range(len(subject_variables))) # I(X;S))

                indiv_mis = [pdf_subjects_snew.mutual_information([var],list(range(len(subject_variables),
                            len(pdf_subjects_snew)))) for var in range(len(subject_variables))] # I(Xi;S)

                # print("P3 FREE,TOT,FRACS", free_params, tot_mi, indiv_mis)
                # print("P3 TOT MI, INDIV MIS", tot_mi, indiv_mis)
                syninfo_naive = tot_mi - sum(indiv_mis)

                # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                cost = (upper_bound_synergistic_information - syninfo_naive) \
                    / upper_bound_synergistic_information
                # add an extra cost term for the fraction of 'individual' information versus the total information
                # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                if tot_mi != 0:
                    cost += sum(indiv_mis) / tot_mi
                else:
                    cost += sum(indiv_mis)
        else:
            print("SUBJECTS IS NONE???")

        # this if-block will add a cost term for not being agnostic to given variables, usually (a) previous SRV(s)
        if not agnostic_about is None:
            assert not subject_variables is None, 'how can all variables be subject_variables and you still want' \
                                                ' to be agnostic about certain (other) variables? (if you did' \
                                                ' not specify subject_variables, do so.)'

            # make a conditional distribution of the synergistic variables conditioned on the subject variables
            # so that I can easily make a new joint pdf object with them and quantify this extra cost for the
            # agnostic constraint

            cond_pdf_syns_on_subjects = pdf_subjects_snew.conditional_probability_distributions(
                range(len(subject_variables))) # pSnew|subjects

            assert type(cond_pdf_syns_on_subjects) == dict \
                or isinstance(cond_pdf_syns_on_subjects, ConditionalProbabilities)

            pXYSSnew = self.copy() # pXYSold
            pXYSSnew.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                                        subject_variables) #pXYSoldSnew
            if np.ndim(agnostic_about) == 1:
                # note: cost term for agnostic is in [0,1]
                
                agn_mis = pXYSSnew.mutual_information(synergistic_variables, agnostic_about)
                upper_bound_agnostic_information = pXYSSnew.entropy(agnostic_about)
                cost += agn_mis / upper_bound_agnostic_information
            else:
                assert np.ndim(agnostic_about) == 2, 'expected list of lists, not more... made a mistake?'

                assert False, 'I don\'t think this case should happen, according to my 2017 paper should be just ' \
                            'I(X:A) so ndim==1'

                upper_bound_agnostic_information = sum([entropy(self,ai) for ai in agnostic_about])
                for agn_i in agnostic_about:
                    # note: total cost term for agnostic is in [0,1]
                    cost += (1.0 / float(len(agnostic_about))) * \
                            pXYSSnew.mutual_information([-1], agn_i) \
                            / upper_bound_agnostic_information

        assert np.isscalar(cost)
        assert np.isfinite(cost)
        # print(cost)
        return float(cost)

    # these options are altered mainly to try to lower the computation time, which is considerable.
    minimize_options = {'ftol': 1e-6}
    optres_list = []
    round_dec = 9
    initial_guess = []

    if multi:
        if initial_guesses:
            initial_guess = [[init] for init in initial_guesses]
        else:
            initial_guess = [[[]] for _ in range(num_repeats)]
        optres,data = append_with_mpi([self,parX, pdf_subjects_snew, subject_variables,\
            agnostic_about, num_free_parameters,upper_bound_synergistic_information],num_repeats,initial_guess,data)
    else:
        for ix in range(num_repeats):
            if not initial_guesses or ix >= len(initial_guesses):
                initial_guess = list(np.round(np.random.random(num_free_parameters),round_dec))
            else:
                initial_guess = initial_guesses[ix]

            sim = Simulator(cost_func_subjects_only)
            time_before = time.time()
        
            optres_ix = minimize(sim.simulate,
                                    initial_guess,
                                    bounds=[(0.0, 1.0)]*num_free_parameters,
                                    callback=sim.callback,
                                #  callback=(lambda xv: param_vectors_trace.append(list(xv))) if verbose else None,
                                    args=(parX,), method=minimize_method,options=minimize_options)
            # print("P3 PARAM X",parX)
            # cost = cost_func_subjects_only(initial_guess,parX)
            # class AttrDict(dict):
            #     def __init__(self, *args,**kwargs):
            #         super(AttrDict,self).__init__(*args,**kwargs)
            #         self.__dict__ = self
            # optres_ix = {"succes":False,"x":initial_guess}
            # optres_ix = AttrDict(optres_ix)
            # data['all_cost'][-1].append(cost)
            data['all_runtimes'][-1].append(time.time()-time_before)           
            data['all_initials'][-1].append(list(initial_guess))
            data['all_finals'][-1].append(list(optres_ix.x))
            data['all_cost'][-1].append(optres_ix.fun)

            # make a conditional distribution of the synergistic variables conditioned on the subject variables
            # so that I can easily make a new joint pdf object with them and quantify this extra cost for the
            # agnostic constraint
            if not tol_nonsyn_mi_frac is None:
                params2matrix_incremental(pdf_subjects_snew,list(parX) + list(optres_ix.x))
                assert not pdf_subjects_snew is None

                tot_mi = pdf_subjects_snew.mutual_information(range(len(subject_variables), len(pdf_subjects_snew)),
                                range(len(subject_variables))) # I(X;S))

                indiv_mis = [pdf_subjects_snew.mutual_information([var],range(len(subject_variables),len(pdf_subjects_snew)))
                                for var in range(len(subject_variables))] # I(Xi;S)
                frac_subjectsSnew = sum(indiv_mis) / float(tot_mi)
                if frac_subjectsSnew > tol_nonsyn_mi_frac:
                    continue  # don't add this to the list of solutions

            data['agn_fracs'].append(-1)
            if not tol_agn_mi_frac is None and not agnostic_about is None:
                if len(agnostic_about) > 0:

                    cond_pdf_syns_on_subjects = pdf_subjects_snew.conditional_probability_distributions(
                        range(len(subject_variables))) # pSnew|subjects

                    assert type(cond_pdf_syns_on_subjects) == dict \
                        or isinstance(cond_pdf_syns_on_subjects, ConditionalProbabilities)

                    pXYSSnew = self.copy() # pXYSold
                    pXYSSnew.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                                                subject_variables)
                    agn_ent = pXYSSnew.entropy(agnostic_about)
                    agn_frac_subjectsSnew = (pXYSSnew.mutual_information(agnostic_about,range(len(self),len(pXYSSnew))))\
                        / agn_ent

                    if agn_frac_subjectsSnew > tol_agn_mi_frac:
                        continue  # don't add this to the list of solutions
                    data['agn_fracs'].append(agn_frac_subjectsSnew)

            optres_list.append(optres_ix)
            data['tries'].append(ix)
            data['srv_paths'].append([[initial_guess]+np.array(sim.list_callback_inp).tolist()])

        optres_list = [resi for resi in optres_list if resi.success]  # filter out the unsuccessful optimizations

        if len(optres_list) == 0:
            raise UserWarning('all ' + str(num_repeats) + ' optimizations using minimize() failed...?!')

        optres_list = np.array([[data['tries'][i],resi] for i,resi in enumerate(optres_list) if resi.success])  # filter out the unsuccessful optimizations
        data['srv_ids'] = list(optres_list[:,0])
        optres_list = list(optres_list[:,1])
        costvals = [res.fun for res in optres_list]
        min_cost = min(costvals)
        optres_ix = costvals.index(min_cost)
        optres = optres_list[optres_ix]
        assert optres_ix >= 0 and optres_ix < len(optres_list)

        # only save entropy data of best srv and params of all found valid srvs
        data['best_n'] = data['srv_ids'][optres_ix]
        assert len(data['srv_paths']) == len(optres_list) # aka resi.success True if free params are valid

    assert len(optres.x) == num_free_parameters
    assert max(optres.x) <= 1.0000001, 'parameter bound significantly violated, ' + str(max(optres.x))
    assert min(optres.x) >= -0.0000001, 'parameter bound significantly violated, ' + str(min(optres.x))

    # todo: reuse the .append_optimized_variables (or so) instead, passing the cost function only? would also
    # validate that function.

    # clip to valid range, add new srv parameters to p(XS)
    optres.x = [min(max(xi, 0.0), 1.0) for xi in optres.x]
    params2matrix_incremental(pdf_subjects_snew,list(parX) + list(optres.x))

    # transform new p(XS) to p(XYS)
    if not subject_variables is None:
        cond_pdf_syns_on_subjects = pdf_subjects_snew.conditional_probability_distributions(
            range(len(subject_variables))
        )
        assert isinstance(cond_pdf_syns_on_subjects, ConditionalProbabilities)
        assert cond_pdf_syns_on_subjects.num_given_variables() > 0, 'conditioned on 0 variables?'

        # if this hits then something is unintuitive with conditioning on variables...
        assert cond_pdf_syns_on_subjects.num_given_variables() == len(subject_variables)

        pdf_with_srvs = self.copy()
        pdf_with_srvs.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                       given_variables=subject_variables)
    else:
        pdf_with_srvs = pdf_subjects_snew  # all variables are subject

    assert pdf_with_srvs.numvariables == self.numvariables + num_synergistic_variables

    self.duplicate(pdf_with_srvs)
      
# todo: return not just a number but an object with more result information, which maybe if evaluated as
# float then it will return the current return value
def synergistic_information(self, variables_Y, variables_X,num_srvs=1,tol_nonsyn_mi_frac=0.05, verbose=False,
                            summed_modulo=False,minimize_method=None, num_repeats_per_srv_append=1,\
                                multi=False,all_initials=[]):
    pdf_with_srvs = self.copy()
    # TODO: improve the retrying of finding and adding srvs, in different orders?
    parX = list(matrix2params_incremental(pdf_with_srvs[variables_X]))
    syn_entropy = synergistic_entropy_upper_bound(self,variables_X)  # H(X1,..,Xn) - max(H(Xi))
    max_ent_per_var = np.log2(self.numvalues)  # max H(S)
    max_num_srv_add_trials = int(round(syn_entropy / max_ent_per_var * 2 + 0.5))  # heuristic
    # max_num_srv_add_trials = 2
    # ent_X = d.entropy(variables_X)

    # get relevant data from synergistic information calculation
    srv_keys = ['best_n','srv_ids','srv_paths','agn_fracs','tries']
    all_keys = ['all_initials','all_finals','all_cost','all_runtimes']
    data = {}
    for k in srv_keys+all_keys:
        data[k] = []
    mutuals = ['I(X;S)','I(Y;S)','mi_fracs']
    final_data = {k:[] for k in list(data.keys())+mutuals+['try']}
    final_data['parXYSold'] = []

    # note: currently I constrain to add SRVs which consist each of only 1 variable each. I guess in most cases
    # this will work fine, but I do not know for sure that this will always be sufficient (i.e., does there
    # ever an MSRV exist with bigger entropy, and if so, is this MSRV not  decomposable into parts such that
    # this approach here still works
    total_syn_mi = 0
    for i in range(max_num_srv_add_trials):
        print("TRIAL ",i)
        for r in data.keys():
            if r in all_keys:
                data[r].append([])
            else:
                data[r] = []

        final_data['parXYSold'].append(list(matrix2params_incremental(pdf_with_srvs)))
        final_data['try'].append(i)
        initial_guess = []
        if all_initials and i < len(all_initials):
            initial_guess = all_initials[i]
        
        try:
            agnostic_about = range(len(self), len(pdf_with_srvs))  # new SRV must not correlate with previous SRVs
            append_synergistic_variables(pdf_with_srvs,parX,data,num_srvs,summed_modulo=summed_modulo,\
                                        subject_variables=variables_X,num_repeats=num_repeats_per_srv_append,\
                                        agnostic_about=agnostic_about,minimize_method=minimize_method,\
                                        tol_nonsyn_mi_frac=tol_nonsyn_mi_frac, tol_agn_mi_frac=tol_nonsyn_mi_frac,\
                                        initial_guesses=initial_guess,multi=multi)
        except UserWarning as e:
            assert 'minimize() failed' in str(e), 'only known reason for this error'
            warnings.warn(str(e) + '. Will now skip this sample in synergistic_information.')
            continue
        
        # np.append(subjects, len(subjects)) 
        total_mi = pdf_with_srvs.mutual_information([-1], variables_X) # I(Snew;X)
        indiv_mi_list = [pdf_with_srvs.mutual_information([-1], [xid]) for xid in variables_X] #sum_i(I(Snew;Xi))
        new_syn_info = total_mi - sum(indiv_mi_list)  # lower bound, actual amount might be higher (but < total_mi)

        if new_syn_info < syn_entropy * 0.01:  # very small added amount if syn. info., so stop after this one
            if (total_mi - new_syn_info) / total_mi > tol_nonsyn_mi_frac:  # too much non-syn. information?
                pdf_with_srvs.marginalize_distribution(range(len(pdf_with_srvs) - 1))  # remove last srv

                if verbose:
                    print('debug: synergistic_information: a last SRV was found but with too much non-syn. info.')
            else:
                if verbose:
                    print('debug: synergistic_information: a last (small) '
                          'SRV was found, a good one, and now I will stop.')

                total_syn_mi += new_syn_info
                final_data['I(X;S)'].append(total_mi)
                final_data['mi_fracs'].append(indiv_mi_list)  
                for d in data.keys():
                    if d not in all_keys:
                        final_data[d].append(data[d])

            break  # assume that no more better SRVs can be found from now on, so stop (save time)
        else:
            if i == max_num_srv_add_trials - 1:
                warnings.warn('synergistic_information: never converged to adding SRV with ~zero synergy')

            if (total_mi - new_syn_info) / total_mi > tol_nonsyn_mi_frac:  # too much non-synergistic information?
                pdf_with_srvs.marginalize_distribution(range(len(pdf_with_srvs) - 1))  # remove last srv

                if verbose:
                    print('debug: synergistic_information: an SRV with new_syn_info/total_mi = '
                          + str(new_syn_info) + ' / ' + str(total_mi) + ' = '
                          + str(new_syn_info / total_mi) + ' was found, which will be removed again because'
                          ' it does not meet the tolerance of ' + str(tol_nonsyn_mi_frac))
            else:
                if verbose:
                    print('debug: synergistic_information: an SRV with new_syn_info/total_mi = '
                          + str(new_syn_info) + ' / ' + str(total_mi) + ' = '
                          + str(new_syn_info / total_mi) + ' was found, which satisfies the tolerance of '
                          + str(tol_nonsyn_mi_frac))

                    if len(agnostic_about) > 0:
                        agn_mi = pdf_with_srvs.mutual_information([-1], agnostic_about)

                        print('debug: agnostic=%s, agn. mi = %s (should be close to 0)' % (agnostic_about, agn_mi))

                total_syn_mi += new_syn_info
                final_data['I(X;S)'].append(total_mi)
                final_data['mi_fracs'].append(indiv_mi_list)  
                for d in data.keys():
                    if d not in all_keys:
                        final_data[d].append(data[d])

        if total_syn_mi >= syn_entropy * 0.99:
            if verbose:
                print('debug: found {} of the upper bound of synergistic entropy which is high so I stop.'
                      % syn_entropy)
            break  # found enough SRVs, cannot find more (except for this small %...)

    if verbose: 
        print('debug: synergistic_information: number of SRVs:', len(range(len(self), len(pdf_with_srvs))))
        print('debug: synergistic_information: entropy of SRVs:',
              pdf_with_srvs.entropy(range(len(self), len(pdf_with_srvs))))
        print('debug: synergistic_information: H(SRV_i) =',
              [pdf_with_srvs.entropy([srv_id]) for srv_id in range(len(self), len(pdf_with_srvs))])
        print('debug: synergistic_information: I(Y, SRV_i) =',
              [pdf_with_srvs.mutual_information(variables_Y, [srv_id])
               for srv_id in range(len(self), len(pdf_with_srvs))])

    Si_mutualY = [pdf_with_srvs.mutual_information(variables_Y, [srv_id])
                    for srv_id in range(len(self), len(pdf_with_srvs))]
    final_data['I(Y;S)'] = Si_mutualY
    for a in all_keys:
        final_data[a] = data[a]
    return final_data

def synergistic_information_naive(self, variables_SRV, variables_X, return_also_relerr=False):
    """
    Estimate the amount of synergistic information contained in the variables (indices) <variables_SRV> about the
    variables (indices) <variables_X>. It is a naive estimate which works best if <variables_SRV> is (approximately)
    an SRV, i.e., if it already has very small MI with each individual variable in <variables_X>.

    Also referred to as the Whole-Minus-Sum (WMS) algorithm.

    Note: this is not compatible with the definition of synergy used by synergistic_information(), i.e., one is
    not an unbiased estimator of the other or anything. Very different.

    :param variables_SRV:
    :param variables_X:
    :param return_also_relerr: if True then a tuple of 2 floats is returned, where the first is the best estimate
        of synergy and the second is the relative error of this estimate (which is preferably below 0.1).
    :rtype: float or tuple of float
    """

    indiv_mis = [self.mutual_information(list(variables_SRV), list([var_xi])) for var_xi in variables_X]
    total_mi = self.mutual_information(list(variables_SRV), list(variables_X))

    syninfo_lowerbound = total_mi - sum(indiv_mis)
    syninfo_upperbound = total_mi - max(indiv_mis)

    if not return_also_relerr:
        return (syninfo_lowerbound + syninfo_upperbound) / 2.0
    else:
        best_estimate_syninfo = (syninfo_lowerbound + syninfo_upperbound) / 2.0
        uncertainty_range = syninfo_upperbound - syninfo_lowerbound

        return (best_estimate_syninfo, uncertainty_range / best_estimate_syninfo)

def append_with_mpi(args,num_repeats,initial_guess,data):
    all_args = [args]*num_repeats
    all_args = [list(itertools.chain(*i)) for i in zip(all_args, initial_guess)]
    with mp.Pool(mp.cpu_count()) as pool:
        data_list = pool.map(append_synvar_innerloop, all_args)
    
    data_list = np.array(data_list)
    data_list,all_optres = data_list[:,0], data_list[:,1]
    
    all_keys = ['all_initials','all_finals','all_cost','all_runtimes']
    for a in all_keys:
        data[a][-1] = [data_list[i][a] for i in range(num_repeats)]
    for b in [i for i in data_list[0].keys() if i not in all_keys+['srv']]:
        data[b] = [data_list[i][b] for i in range(num_repeats)]
    data['srv_ids'] = [i for i, x in enumerate([data_list[i]['srv'] for i in range(num_repeats)]) if x]

    optres_list = []
    cost_list = []
    for idx in data['srv_ids']:
        optres_list.append(all_optres[idx])
        cost_list.append(data['all_cost'][-1][idx])

    if len(optres_list) == 0:
        raise UserWarning('all ' + str(num_repeats) + ' optimizations using minimize() failed...?!')

    min_cost = min(cost_list)
    optres_ix = cost_list.index(min_cost)
    optres = optres_list[optres_ix]
    data['best_n'] = optres_ix
    return optres,data

def append_synvar_innerloop(args, minimize_method=None, minimize_options=None,
                            tol_nonsyn_mi_frac=0.05, tol_agn_mi_frac=0.05):
    self, parameter_values_static, pdf_subjects_snew, subject_variables,agnostic_about\
        ,num_free_parameters_synonly,upper_bound_synergistic_information,initial_guess = args
    num_synergistic_variables = 1
    data = {}
    if not initial_guess:
        round_dec = 9
        initial_guess = list(np.round(np.random.random(num_free_parameters_synonly),round_dec))

    def cost_func_subjects_only_mpi(free_params, parameter_values_before, extra_cost_rel_error=True):
        """
        This cost function searches for a Pr(S,Y,A,X) such that X is synergistic about S (subject_variables) only.
        This fitness function also taxes any correlation between X and A (agnostic_variables), but does not care
        about the relation between X and Y.
        :param free_params:
        :param parameter_values_before:
        :return:
        """
        if min(free_params) < -0.00001 or max(free_params) > 1.00001:
            warnings.warn('scipy\'s minimize() is violating the parameter bounds 0...1 I give it: '
                        + str(free_params))

            # high cost for invalid parameter values
            # note: maximum cost normally from this function is about 2.0
            return 10.0 + 100.0 * np.sum([p - 1.0 for p in free_params if p > 1.0]
                                        + [np.abs(p) for p in free_params if p < 0.0])

        # assert max(free_params) <= 1.00001, \
        #     'scipy\'s minimize() is violating the parameter bounds 0...1 I give it: ' + str(free_params)

        free_params = [min(max(fp, 0.0), 1.0) for fp in free_params]  # clip small roundoff errors
        params2matrix_incremental(pdf_subjects_snew,list(parameter_values_before) + list(free_params))

        # make a conditional distribution of the synergistic variables conditioned on the subject variables
        # so that I can easily make a new joint pdf object with them and quantify this extra cost for the
        # agnostic constraint
        cond_pdf_syns_on_subjects = pdf_subjects_snew.conditional_probability_distributions(
            range(len(subject_variables))) # pSnew|subjects

        assert type(cond_pdf_syns_on_subjects) == dict \
            or isinstance(cond_pdf_syns_on_subjects, ConditionalProbabilities)

        pXYSSnew = self.copy() # pXYSold
        pXYSSnew.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                                    subject_variables) #pXYSoldSnew

        tot_mi = pdf_subjects_snew.mutual_information(range(len(subject_variables), len(pdf_subjects_snew)),
                        range(len(subject_variables))) # I(X;S))

        indiv_mis = [pdf_subjects_snew.mutual_information([var],range(len(subject_variables),len(pdf_subjects_snew)))
                        for var in range(len(subject_variables))] # I(Xi;S)

        syninfo_naive = tot_mi - sum(indiv_mis)

        # this if-block will add cost for the estimated amount of synergy induced by the proposed parameters,
        # and possible also a cost term for the ratio of synergistic versus non-synergistic info as extra
        if not subject_variables is None:
            assert pdf_subjects_snew.numvariables == len(subject_variables) + num_synergistic_variables

            # this can be considered to be in range [0,1] although particularly bad solutions can go >1
            if not extra_cost_rel_error:
                cost = (upper_bound_synergistic_information - pdf_subjects_snew.synergistic_information_naive(
                    variables_SRV=range(len(subject_variables), len(pdf_subjects_snew)),
                    variables_X=range(len(subject_variables)))) / upper_bound_synergistic_information
            else:

                # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                cost = (upper_bound_synergistic_information - syninfo_naive) \
                    / upper_bound_synergistic_information
                # add an extra cost term for the fraction of 'individual' information versus the total information
                # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                if tot_mi != 0:
                    cost += sum(indiv_mis) / tot_mi
                else:
                    cost += sum(indiv_mis)

        # this if-block will add a cost term for not being agnostic to given variables, usually (a) previous SRV(s)
        if not agnostic_about is None:
            assert not subject_variables is None, 'how can all variables be subject_variables and you still want' \
                                                ' to be agnostic about certain (other) variables? (if you did' \
                                                ' not specify subject_variables, do so.)'

            if np.ndim(agnostic_about) == 1:
                # note: cost term for agnostic is in [0,1]
                
                agn_mis = pXYSSnew.mutual_information([-1], agnostic_about)
                upper_bound_agnostic_information = pXYSSnew.entropy(agnostic_about)
                cost += agn_mis / upper_bound_agnostic_information
            else:
                assert np.ndim(agnostic_about) == 2, 'expected list of lists, not more... made a mistake?'

                assert False, 'I don\'t think this case should happen, according to my 2017 paper should be just ' \
                            'I(X:A) so ndim==1'

                upper_bound_agnostic_information = sum([entropy(self,ai) for ai in agnostic_about])
                for agn_i in agnostic_about:
                    # note: total cost term for agnostic is in [0,1]
                    cost += (1.0 / float(len(agnostic_about))) * \
                            pXYSSnew.mutual_information([-1], agn_i) \
                            / upper_bound_agnostic_information

        assert np.isscalar(cost)
        assert np.isfinite(cost)
        return float(cost)

    sim = Simulator(cost_func_subjects_only_mpi)
    time_before = time.time()
    optres_ix = minimize(sim.simulate,initial_guess,
                        args=(parameter_values_static,),
                        bounds=[(0, 1.0)]*num_free_parameters_synonly, method=minimize_method, options=minimize_options)

    data['all_runtimes']=time.time()-time_before
    data['all_initials']=list(initial_guess)
    data['all_finals']=list(optres_ix.x)
    data['all_cost']=optres_ix.fun

    # make a conditional distribution of the synergistic variables conditioned on the subject variables
    # so that I can easily make a new joint pdf object with them and quantify this extra cost for the
    # agnostic constraint
    if not tol_nonsyn_mi_frac is None:
        params2matrix_incremental(pdf_subjects_snew,list(parameter_values_static) + list(optres_ix.x))
        assert not pdf_subjects_snew is None

        tot_mi = pdf_subjects_snew.mutual_information(range(len(subject_variables), len(pdf_subjects_snew)),
                        range(len(subject_variables))) # I(X;S))

        indiv_mis = [pdf_subjects_snew.mutual_information([var],range(len(subject_variables),len(pdf_subjects_snew)))
                        for var in range(len(subject_variables))] # I(Xi;S)
        frac_subjectsSnew = sum(indiv_mis) / float(tot_mi)
    data['agn_fracs']=-1
    
    srv = False
    if frac_subjectsSnew < tol_nonsyn_mi_frac:
        srv = True
        if not tol_agn_mi_frac is None and not agnostic_about is None:
            if len(agnostic_about) > 0:

                cond_pdf_syns_on_subjects = pdf_subjects_snew.conditional_probability_distributions(
                    range(len(subject_variables))) # pSnew|subjects

                assert type(cond_pdf_syns_on_subjects) == dict \
                    or isinstance(cond_pdf_syns_on_subjects, ConditionalProbabilities)

                pXYSSnew = self.copy() # pXYSold
                pXYSSnew.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                                            subject_variables)
                agn_ent = pXYSSnew.entropy(agnostic_about)
                agn_frac_subjectsSnew = (pXYSSnew.mutual_information(agnostic_about,range(len(self),len(pXYSSnew))))\
                    / agn_ent

                if agn_frac_subjectsSnew > tol_agn_mi_frac:
                    srv = False
                else:
                    data['agn_fracs']=agn_frac_subjectsSnew            

    data['srv'] = srv
    if srv:
        data['srv_paths'] = [initial_guess]+np.array(sim.list_callback_inp).tolist()
    else:
        data['srv_paths'] = []

    return [data, optres_ix]
