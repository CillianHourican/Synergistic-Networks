import itertools
import numpy as np
import warnings
import multiprocessing as mp
import string
import copy
import dit
from abc import abstractmethod, ABCMeta
from numbers import Number
from numbers import Integral
_type_prob = np.float


from utils import maximum_depth, flatten, get_var_labels, apply_permutation

from sys import platform
if platform == "linux" or platform == "linux2":
    # linux
    _type_prob = np.float128
elif platform == "darwin":
    # OS X
    _type_prob = np.float128
elif platform == "win32":
    # Windows...
    _type_prob = np.float

_default_variable_label = 'variable'

##############################################################################
#   FullNestedArrayOfProbabilities
##############################################################################
class NestedArrayOfProbabilities(object):

    __metaclass__ = ABCMeta

    # type: np.array
    joint_probabilities = np.array([], dtype=_type_prob)

    @abstractmethod
    def __init__(self, joint_probabilities=np.array([], dtype=_type_prob)):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def num_variables(self):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def num_values(self):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def __getitem__(self, item):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def __setitem__(self, key, value):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def sum(self):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def flatiter(self):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def clip_all_probabilities(self):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def flatten(self):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def reset(self, jointprobs):  # basically __init__ but can be called even if object already exists
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def duplicate(self, other):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def __sub__(self, other):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def generate_uniform_joint_probabilities(self, numvariables, numvalues):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def generate_random_joint_probabilities(self, numvariables, numvalues):
        assert False, 'must be implemented by subclass'

    @abstractmethod
    def generate_dirichlet_joint_probabilities(self, numvariables, numvalues):
        assert False, 'must be implemented by subclass'


class FullNestedArrayOfProbabilities(NestedArrayOfProbabilities):
    # type: np.array
    joint_probabilities = np.array([], dtype=_type_prob)

    def __init__(self, joint_probabilities=np.array([], dtype=_type_prob)):
        # assert isinstance(joint_probabilities, np.ndarray), 'should pass numpy array?'

        self.reset(joint_probabilities)

    def num_variables(self):
        return np.ndim(self.joint_probabilities)

    def num_values(self):
        return np.shape(self.joint_probabilities)[-1]

    def __getitem__(self, item):
        assert len(item) == self.joint_probabilities.ndim, 'see if this is only used to get single joint probs'

        if type(item) == int:
            return FullNestedArrayOfProbabilities(joint_probabilities=self.joint_probabilities[item])
        elif hasattr(item, '__iter__'):  # tuple for instance
            ret = self.joint_probabilities[item]

            if hasattr(ret, '__iter__'):
                return FullNestedArrayOfProbabilities(joint_probabilities=ret)
            else:
                assert -0.000001 <= ret <= 1.000001  # this seems the only used case

                return ret  # return a single probability
        elif isinstance(item, slice):
            return FullNestedArrayOfProbabilities(joint_probabilities=self.joint_probabilities[item])

    def __setitem__(self, key, value):
        assert False, 'see if used at all?'

        # let numpy array figure it out
        self.joint_probabilities[key] = value

    def sum(self):
        return self.joint_probabilities.sum()

    def flatiter(self):
        return self.joint_probabilities.flat

    def clip_all_probabilities(self):
        """
        Make sure all probabilities in the joint probability matrix are in the range [0.0, 1.0], which could be
        violated sometimes due to floating point operation roundoff errors.
        """
        self.joint_probabilities = np.array(np.minimum(np.maximum(self.joint_probabilities, 0.0), 1.0),
                                            dtype=_type_prob)

        if np.random.random() < 0.05:
            try:
                np.testing.assert_almost_equal(np.sum(self.joint_probabilities), 1.0)
            except AssertionError as e:
                print('error message: ' + str(e))

                print('error: len(self.joint_probabilities) =', len(self.joint_probabilities))
                print('error: shape(self.joint_probabilities) =', np.shape(self.joint_probabilities))
                if len(self.joint_probabilities) < 30:
                    print('error: self.joint_probabilities =', self.joint_probabilities)

                raise AssertionError(e)

    def flatten(self):
        return self.joint_probabilities.flatten()

    def reset(self, jointprobs):  # basically __init__ but can be called even if object already exists
        if isinstance(jointprobs, np.ndarray):
            self.joint_probabilities = np.array(jointprobs, dtype=_type_prob)
        else:
            # assert isinstance(jointprobs.joint_probabilities, np.ndarray), 'nested reset problem'

            self.duplicate(jointprobs)

    def duplicate(self, other):
        assert isinstance(other.joint_probabilities, np.ndarray), 'nesting problem'

        self.joint_probabilities = np.array(other.joint_probabilities, dtype=_type_prob)

    def __sub__(self, other):
        return np.subtract(self.joint_probabilities, np.array(other.joint_probabilities, dtype=_type_prob))

    def generate_uniform_joint_probabilities(self, numvariables, numvalues):
        self.joint_probabilities = np.array(np.zeros([numvalues]*numvariables), dtype=_type_prob)
        self.joint_probabilities = self.joint_probabilities + 1.0 / np.power(numvalues, numvariables)

    def generate_random_joint_probabilities(self, numvariables, numvalues):
        # todo: this does not result in random probability densities... Should do recursive
        self.joint_probabilities = np.random.random([numvalues]*numvariables)
        self.joint_probabilities /= np.sum(self.joint_probabilities)

    def generate_dirichlet_joint_probabilities(self, numvariables, numvalues):
        # todo: this does not result in random probability densities... Should do recursive
        self.joint_probabilities = np.random.dirichlet([1]*(numvalues**numvariables)).reshape((numvalues,)*numvariables)

# todo: move params2matrix and matrix2params to the NestedArray classes? Is specific per subclass...
class IndependentNestedArrayOfProbabilities(NestedArrayOfProbabilities):
    __metaclass__ = ABCMeta
    """
    note: each sub-array sums to 1.0
    type: np.array of np.array of float
    """
    marginal_probabilities = np.array([])  # this initial value indicates: zero variables

    def __init__(self, joint_probabilities=np.array([], dtype=_type_prob)):
        self.reset(joint_probabilities)

    def num_variables(self):
        """
        The number of variables for which this object stores a joint pdf.
        """
        return len(self.marginal_probabilities)

    def num_values(self):
        assert len(self.marginal_probabilities) > 0, 'not yet initialized, so cannot determine num_values'

        return len(self.marginal_probabilities[0])

    def __getitem__(self, item):
        """
        If you supply an integer then it specifies the value of the first variable, so I will return the remaining
        PDF for the rest of the variables. If you supply a tuple then I will repeat this for every integer in the
        tuple.
        :param item: value or tuple of values for the first N variables, where N is the length of <item>.
        :type item: int or tuple
        """
        if isinstance(item, Integral):
            assert False, 'seems not used?'

            return IndependentNestedArrayOfProbabilities(self.marginal_probabilities[0][item]
                                                         * self.marginal_probabilities[len(item):])
        else:
            if len(item) == len(self.marginal_probabilities):
                return np.product([self.marginal_probabilities[vix][item[vix]] for vix in range(len(item))])
            else:
                assert len(item) < len(self.marginal_probabilities), 'supplied more values than I have variables'

                return IndependentNestedArrayOfProbabilities(np.product([self.marginal_probabilities[vix][item[vix]]
                                                                         for vix in range(len(item))])
                                                             * self.marginal_probabilities[len(item):])

    def __setitem__(self, key, value):
        assert False, 'is this used? if so, make implementation consistent with getitem? So that getitem(setitem)' \
                      ' yields back <value>? Now thinking of it, I don\'t think this is possible, because how would' \
                      ' I set the value of a joint state? It is made up of a multiplication of |X| probabilities,' \
                      ' and setting these probabilities to get the supplied joint probability is an ambiguous task.'

    def sum(self):
        """
        As far as I know, this is only used to verify that probabilities sum up to 1.
        """
        return np.average(np.sum(self.marginal_probabilities, axis=1))

    def flatiter(self):
        return self.marginal_probabilities.flat

    def clip_all_probabilities(self):
        self.marginal_probabilities = np.minimum(np.maximum(self.marginal_probabilities, 0.0), 1.0)

    def flatten(self):
        return self.marginal_probabilities.flatten()

    def reset(self, joint_probabilities):  # basically __init__ but can be called even if object already exists
        if isinstance(joint_probabilities, np.ndarray):
            shape = np.shape(joint_probabilities)

            # assume that the supplied array is in MY format, so array of single-variable prob. arrays
            if len(shape) == 2:
                assert joint_probabilities[0].sum() <= 1.000001, 'should be (marginal) probabilities'

                self.marginal_probabilities = np.array(joint_probabilities, dtype=_type_prob)
            elif len(joint_probabilities) == 0:
                # zero variables
                self.marginal_probabilities = np.array([], dtype=_type_prob)
            else:
                raise ValueError('if you want to supply a nested array of probabilities, create a joint pdf'
                                 ' from it first and then pass it to me, because I would need to marginalize variables')
        elif isinstance(joint_probabilities, JointProbabilityMatrix):
            # TODO: Maybe move matrix2vector into standalone function
            self.marginal_probabilities = np.array([joint_probabilities.marginalize_distribution([vix])
                                                   .joint_probabilities.matrix2vector()
                                                    for vix in range(len(joint_probabilities))], dtype=_type_prob)
        else:
            raise NotImplementedError('unknown type of argument: ' + str(joint_probabilities))

    def duplicate(self, other):
        """
        Become a copy of <other>.
        :type other: IndependentNestedArrayOfProbabilities
        """
        self.marginal_probabilities = np.array(other.marginal_probabilities, dtype=_type_prob)

    def __sub__(self, other):
        """
        As far as I can tell, only needed to test e.g. for np.testing.assert_array_almost_equal.
        :type other: IndependentNestedArrayOfProbabilities
        """
        return self.marginal_probabilities - other.marginal_probabilities

    def generate_uniform_joint_probabilities(self, numvariables, numvalues):
        self.marginal_probabilities = np.array(np.zeros([numvariables, numvalues]), dtype=_type_prob)
        self.marginal_probabilities = self.marginal_probabilities + 1.0 / numvalues

    def generate_random_joint_probabilities(self, numvariables, numvalues):
        self.marginal_probabilities = np.array(np.random.random([numvariables, numvalues]), dtype=_type_prob)
        # normalize:
        self.marginal_probabilities /= np.transpose(np.tile(np.sum(self.marginal_probabilities, axis=1), (numvalues,1)))


##############################################################################
#   ConditionalProbabilities & ConditionalProbabilityMatrix
##############################################################################
class ConditionalProbabilities(object):

    # for enabling the decorator 'abstractmethod', which means that a given function MUST be overridden by a
    # subclass. I do this because all functions depend heavily on the data structure of cond_pdf, but derived
    # classes will highly likely have different structures, because that would be the whole point of inheritance:
    # currently the cond_pdf encodes the full conditional pdf using a dictionary, but the memory usage of that
    # scales combinatorially.
    __metaclass__ = ABCMeta

    # a dict which fully explicitly encodes the conditional pdf,
    # namely, dict where keys are
    # tuples of values for all variables and the values are JointProbabilityMatrix objects
    cond_pdf = {}

    @abstractmethod
    def __init__(self, cond_pdf=None):
        assert False, 'should have been implemented'

    @abstractmethod
    def generate_random_conditional_pdf(self, num_given_variables, num_output_variables, num_values=2):
        assert False, 'should have been implemented'

    @abstractmethod
    def __getitem__(self, item):
        assert False, 'should have been implemented'

    @abstractmethod
    def iteritems(self):
        assert False, 'should have been implemented'

    @abstractmethod
    def itervalues(self):
        assert False, 'should have been implemented'

    @abstractmethod
    def iterkeys(self):
        assert False, 'should have been implemented'

    @abstractmethod
    def num_given_variables(self):
        assert False, 'should have been implemented'

    @abstractmethod
    def num_output_variables(self):
        assert False, 'should have been implemented'

    @abstractmethod
    def num_conditional_probabilities(self):
        assert False, 'should have been implemented'

    @abstractmethod
    def update(self, partial_cond_pdf):
        assert False, 'should have been implemented'


class ConditionalProbabilityMatrix(ConditionalProbabilities):

    # a dict which fully explicitly encodes the conditional pdf,
    # namely, dict where keys are
    # tuples of values for all variables and the values are JointProbabilityMatrix objects
    cond_pdf = {}

    def __init__(self, cond_pdf=None):
        if cond_pdf is None:
            self.cond_pdf = {}
        elif type(cond_pdf) == dict:
            assert not np.isscalar(list(cond_pdf.keys())[0]), 'should be tuples, even if conditioning on 1 var'

            self.cond_pdf = cond_pdf
        elif isinstance(cond_pdf, JointProbabilityMatrix):
            self.cond_pdf = {states: cond_pdf for states in cond_pdf.statespace()}
        else:
            raise NotImplementedError('unknown type for cond_pdf')

    def __getitem__(self, item):
        return self.cond_pdf[item]

    def __len__(self):
        return len(self.cond_pdf)

    def __eq__(self, other):
        if isinstance(other, str):
            return False
        elif isinstance(other, Number):
            return False
        else:
            assert hasattr(other, 'num_given_variables'), 'should be ConditionalProbabilityMatrix'
            assert hasattr(other, 'iteritems'), 'should be ConditionalProbabilityMatrix'

            for states, pdf in self.iteritems():
                if not other[states] == pdf:
                    return False

            return True

    def generate_random_conditional_pdf(self, num_given_variables, num_output_variables, num_values=2):
        pdf = JointProbabilityMatrix(num_given_variables, num_values)  # only used for convenience, for .statespace()

        self.cond_pdf = {states: JointProbabilityMatrix(num_output_variables, num_values)
                         for states in pdf.statespace()}

    def __getitem__(self, item):
        assert len(self.cond_pdf) > 0, 'uninitialized dict'

        return self.cond_pdf[item]

    def iteritems(self):
        return self.cond_pdf.items()

    def itervalues(self):
        return self.cond_pdf.itervalues()

    def iterkeys(self):
        return self.cond_pdf.iterkeys()

    def num_given_variables(self):
        assert len(self.cond_pdf) > 0, 'uninitialized dict'

        return len(list(self.cond_pdf.keys())[0])

    def num_output_variables(self):
        assert len(self.cond_pdf) > 0, 'uninitialized dict'

        return len(list(self.cond_pdf.values())[0])

    def num_conditional_probabilities(self):
        return len(self.cond_pdf)

    def update(self, partial_cond_pdf):
        if type(partial_cond_pdf) == dict:
            # check only if already initialized with at least one conditional probability
            if __debug__ and len(self.cond_pdf) > 0:
                assert len(list(partial_cond_pdf.keys())[0]) == self.num_given_variables(), 'partial cond. pdf is ' \
                                                                                            'conditioned on a different' \
                                                                                            ' number of variables'
                assert len(next(iter(partial_cond_pdf.values()))) == self.num_output_variables(), \
                    'partial cond. pdf has a different number of output variables'

            self.cond_pdf.update(partial_cond_pdf)
        elif isinstance(partial_cond_pdf, ConditionalProbabilities):
            self.cond_pdf.update(partial_cond_pdf.cond_pdf)
        else:
            raise NotImplementedError('unknown type for partial_cond_pdf')


##############################################################################
#   JointProbabilityMatrix
##############################################################################
class JointProbabilityMatrix(object):
    # nested list of probabilities. For instance for three binary variables it could be:
    # [[[0.15999394, 0.06049343], [0.1013956, 0.15473886]], [[ 0.1945649, 0.15122334], [0.11951818, 0.05807175]]].
    joint_probabilities = FullNestedArrayOfProbabilities()
    numvariables = 0
    numvalues = 0

    # note: at the moment I only store the labels here as courtesy, but the rest of the functions do not operate
    # on labels at all, only variable indices 0..N-1.
    labels = []

    type_prob = _type_prob
    
    def __init__(self, numvariables, numvalues, joint_probs='dirichlet', labels=None,
                 create_using=FullNestedArrayOfProbabilities):

        self.numvariables = numvariables
        self.numvalues = numvalues

        if labels is None:
            self.labels = [_default_variable_label]*numvariables
        else:
            self.labels = labels

        if joint_probs is None or numvariables == 0:
            self.joint_probabilities = create_using()
            self.generate_random_joint_probabilities()
        elif isinstance(joint_probs, str):
            if joint_probs == 'independent' or joint_probs == 'iid' or joint_probs == 'uniform':
                self.joint_probabilities = create_using()
                self.generate_uniform_joint_probabilities(numvariables, numvalues)
            elif joint_probs == 'random':
                # this is BIASED! I.e., if generating bits then the marginal p of one bit being 1 is not uniform
                self.joint_probabilities = create_using()
                self.generate_random_joint_probabilities()
            elif joint_probs in ('unbiased', 'dirichlet'):
                self.joint_probabilities = create_using()
                self.generate_dirichlet_joint_probabilities()  # just to get valid params and pass the debugging tests
            else:
                raise ValueError('don\'t know what to do with joint_probs=' + str(joint_probs))
        else:
            self.joint_probabilities = create_using(joint_probs)

        self.clip_all_probabilities()

        if self.numvariables > 0:
            # if self.numvariables > 1:
            #     assert len(self.joint_probabilities[0]) == numvalues, 'self.joint_probabilities[0] = ' \
            #                                                           + str(self.joint_probabilities[0])
            #     assert len(self.joint_probabilities[-1]) == numvalues
            # else:
            #     assert len(self.joint_probabilities) == numvalues
            #     assert np.isscalar(self.joint_probabilities[(0,)]), 'this is how joint_probabilities may be called'
            
            assert self.joint_probabilities.num_values() == self.numvalues
            assert self.joint_probabilities.num_variables() == self.numvariables
            if __debug__:
                np.testing.assert_almost_equal(self.joint_probabilities.sum(), 1.0)
        else:
            pass
            # warnings.warn('numvariables == 0, not sure if it is supported by all code (yet)! Fingers crossed.')

    
    def __getitem__(self, item):
        if item == 'all':
            return self
        elif not hasattr(item, '__iter__'):
            return self.marginalize_distribution([item])
        else:
            return self.marginalize_distribution(item)
    
    def __sub__(self, other):
        assert len(self) >= len(other), 'cannot compute a conditional pdf consisting of a negative number of variables'

        if len(self) == len(other):
            assert self[range(len(other))] == other, 'my first ' + str(len(other)) + ' variables are not the same as ' \
                                                                                     'that of \'other\''

            return JointProbabilityMatrix(0, self.numvalues)  # return empty pdf...
        elif len(self) > len(other):
            assert self[range(len(other))] == other, 'my first ' + str(len(other)) + ' variables are not the same as ' \
                                                                                     'that of \'other\''

            return self.conditional_probability_distributions(range(len(other)))
        else:
            raise ValueError('len(self) < len(other), '
                             'cannot compute a conditional pdf consisting of a negative number of variables')
    
    def __eq__(self, other):  # approximate to 7 decimals
        if self.numvariables != other.numvariables or self.numvalues != other.numvalues:
            return False
        else:
            try:
                np.testing.assert_array_almost_equal(self.joint_probabilities, other.joint_probabilities)

                return True
            except AssertionError as e:
                assert 'not almost equal' in str(e), 'don\'t know what other assertion could have failed'

                return False

    # todo: implement __setitem__ for either pdf or cond. pdfs
    
    def __add__(self, other):
        """
        Append the variables defined by the (conditional) distributions in other.
        :type other: dict of JointProbabilityMatrix | JointProbabilityMatrix
        :rtype: JointProbabilityMatrix
        """

        pdf = self.copy()
        pdf.append_variables_using_conditional_distributions(other)

        return pdf
    
    def __call__(self, values):
        """
        Joint probability of a list or tuple of values, one value for each variable in order.
        :param values: list of values, each value is an integer in [0, numvalues)
        :type values: list
        :return: joint probability of the given values, in order, for all variables; in [0, 1]
        :rtype: float
        """
        return self.joint_probability(values=values)
    
    def __len__(self):
        return self.numvariables
    
    def copy(self):
        """
        :rtype: TemporalJointProbabilityMatrix
        """
        return copy.deepcopy(self)

    
    def generate_sample(self):
        rand_real = np.random.random()

        for input_states in self.statespace(self.numvariables):
            rand_real -= self.joint_probability(input_states)

            if rand_real < 0.0:
                return input_states

        assert False, 'should not happen? or maybe foating point roundoff errors built up.. or so'

        return self.statespace(len(self.numvariables))[-1]

    
    def generate_samples(self, n, nprocs=1, method='fast'):
        """
        Generate samples from the distribution encoded by this PDF object.
        :param n: number of samples
        :param nprocs: since this procedure is quite slow, from about >=1e5 samples you may want parallel
        :return: list of tuples, shape (n, numvariables)
        :rtype: list
        """
        if nprocs == 1:
            return [self.generate_sample() for _ in range(n)]
        elif method == 'slow':
            def worker_gs(i):
                return self.generate_sample()

            pool = mp.Pool(nprocs)

            ret = pool.map(worker_gs, range(n))

            pool.close()
            pool.terminate()

            return ret
        elif method == 'fast':
            input_states_list = [input_states for input_states in self.statespace(self.numvariables)]

            state_probs = np.array([self.joint_probability(input_states)
                                    for iix, input_states in enumerate(input_states_list)])

            if nprocs == 1:
                state_ixs = np.random.choice(len(input_states_list), size=n, p=state_probs)
            else:
                def worker_gs(numsamples):
                    return np.random.choice(len(input_states_list), size=numsamples, p=np.array(state_probs,
                                                                                                dtype=np.float64))

                numsamples_list = [int(round((pi+1)/float(nprocs) * n)) - int(round(pi/float(nprocs) * n))
                                   for pi in range(nprocs)]

                pool = mp.Pool(nprocs)

                ret = pool.map(worker_gs, numsamples_list)

                pool.close()
                pool.terminate()

                state_ixs = list(itertools.chain.from_iterable(ret))

            return np.array(input_states_list)[state_ixs]
        else:
            raise NotImplementedError('don\'t know this method')

    
    def to_dit(self, base="linear"):
        probs = self.joint_probabilities.joint_probabilities
        # Assume variables with same alphabet/values
        n_vars, vals = len(probs.shape), probs.shape[0]
        labels= get_var_labels(n_vars)
        states = [''.join(i) for i in itertools.product(string.digits[:vals], repeat=n_vars)]
        dist = probs.flatten()
        assert len(states) == len(dist)
        d = dit.Distribution(states, pmf=dist,base=base)
        d.set_rv_names(labels)
        return d
    
    def reset(self, numvariables, numvalues, joint_prob_matrix, labels=None):
        """
        This function is intended to completely reset the object, so if you add variables which determine the
        behavior of the object then add them also here and everywhere where called.
        :type numvariables: int
        :type numvalues: int
        :type joint_prob_matrix: NestedArrayOfProbabilities
            """

        self.numvariables = numvariables
        self.numvalues = numvalues
        if type(joint_prob_matrix) == np.ndarray:
            self.joint_probabilities.reset(np.array(joint_prob_matrix, dtype=self.type_prob))
        else:
            self.joint_probabilities.reset(joint_prob_matrix)

        if labels is None:
            self.labels = [_default_variable_label]*numvariables
        else:
            self.labels = labels

        # assert np.ndim(joint_prob_matrix) == self.numvariables, 'ndim = ' + str(np.ndim(joint_prob_matrix)) + ', ' \
        #                                      'self.numvariables = ' + str(self.numvariables) + ', ' \
        #                                      'joint matrix = ' + str(joint_prob_matrix)
        assert self.joint_probabilities.num_variables() == self.numvariables, \
            'ndim = ' + str(np.ndim(joint_prob_matrix)) + ', ' \
                                                          'self.numvariables = ' + str(self.numvariables) + ', ' \
                                                                                                            'joint matrix = ' + str(self.joint_probabilities.joint_probabilities) \
            + ', joint_probabilities.num_variables() = ' + str(self.joint_probabilities.num_variables())
        assert self.joint_probabilities.num_values() == self.numvalues

        # # maybe this check should be removed, it is also checked in clip_all_* below, but after clipping, which
        # # may be needed to get this condition valid again?
        # if np.random.random() < 0.01:  # make less frequent
        #     np.testing.assert_array_almost_equal(np.sum(joint_prob_matrix), 1.0)

        self.clip_all_probabilities()
    
    def duplicate(self, other_joint_pdf):
        """

        :type other_joint_pdf: JointProbabilityMatrix
        """
        self.reset(other_joint_pdf.numvariables, other_joint_pdf.numvalues, other_joint_pdf.joint_probabilities)
    
    def statespace(self, numvars='all'):
        if numvars == 'all':
            lists_of_possible_joint_values = [range(self.numvalues) for _ in range(self.numvariables)]
        elif type(numvars) in (int,):
            lists_of_possible_joint_values = [range(self.numvalues) for _ in range(numvars)]
        else:
            raise NotImplementedError('numvars=' + str(numvars))
        return itertools.product(*lists_of_possible_joint_values)
    
    def clip_all_probabilities(self):
        """
        Make sure all probabilities in the joint probability matrix are in the range [0.0, 1.0], which could be
        violated sometimes due to floating point operation roundoff errors.
        """
        self.joint_probabilities.clip_all_probabilities()
    
    def joint_probability(self, values):
        assert len(values) == self.numvariables, 'should specify one value per variable'

        if len(self) == 0:
            if len(values) == 0:
                return 1
            else:
                return 0
        else:
            assert values[0] < self.numvalues, 'variable can only take values 0, 1, ..., <numvalues - 1>: ' + str(
                values[0])
            assert values[-1] < self.numvalues, 'variable can only take values 0, 1, ..., <numvalues - 1>: ' \
                                                + str(values[-1])

            joint_prob = self.joint_probabilities[tuple(values)]

            assert 0.0 <= joint_prob <= 1.0, 'not a probability? ' + str(joint_prob)

            return joint_prob
    
    def generate_uniform_joint_probabilities(self, numvariables, numvalues, create_using=None):
        # jp = np.zeros([self.numvalues]*self.numvariables)
        # jp = jp + 1.0 / np.power(self.numvalues, self.numvariables)
        # np.testing.assert_almost_equal(np.sum(jp), 1.0)

        if not create_using is None:
            self.joint_probabilities = create_using()

        self.joint_probabilities.generate_uniform_joint_probabilities(numvariables, numvalues)
    
    def generate_random_joint_probabilities(self, create_using=None):
        # jp = np.random.random([self.numvalues]*self.numvariables)
        # jp /= np.sum(jp)

        if not create_using is None:
            self.joint_probabilities = create_using()

        self.joint_probabilities.generate_random_joint_probabilities(self.numvariables, self.numvalues)
    
    def generate_dirichlet_joint_probabilities(self, create_using=None):
        if not create_using is None:
            self.joint_probabilities = create_using()

        self.joint_probabilities.generate_dirichlet_joint_probabilities(self.numvariables, self.numvalues)
    
    def marginalize_distribution(self, retained_variables):
        """
        Return a pdf of only the given variable indices, summing out all others
        :param retained_variables: variables to retain, all others will be summed out and will not be a variable anymore
        :type: array_like
        :rtype : JointProbabilityMatrix
        """
        lists_of_possible_states_per_variable = [range(self.numvalues) for variable in range(self.numvariables)]

        assert hasattr(retained_variables, '__len__'), 'should be list or so, not int or other scalar'

        marginalized_joint_probs = np.zeros([self.numvalues]*len(retained_variables))

        # if len(variables):
        #     marginalized_joint_probs = np.array([marginalized_joint_probs])

        if len(retained_variables) == 0:
            return JointProbabilityMatrix(0, self.numvalues)

        assert len(retained_variables) > 0, 'makes no sense to marginalize 0 variables'
        assert np.all(map(np.isscalar, retained_variables)), 'each variable identifier should be int in [0, numvalues)'
        assert len(retained_variables) <= self.numvariables, 'cannot marginalize more variables than I have'
        assert len(set(retained_variables)) <= self.numvariables, 'cannot marginalize more variables than I have'

        # not sure yet about doing this:
        # variables = sorted(list(set(variables)))  # ensure uniqueness?

        if np.all(sorted(list(set(retained_variables))) == range(self.numvariables)):
            return self.copy()  # you ask all variables back so I have nothing to do
        else:
            for values in itertools.product(*lists_of_possible_states_per_variable):
                marginal_values = [values[varid] for varid in retained_variables]

                marginalized_joint_probs[tuple(marginal_values)] += self.joint_probability(values)

            if __debug__:
                np.testing.assert_almost_equal(np.sum(marginalized_joint_probs), 1.0)

            marginal_joint_pdf = JointProbabilityMatrix(len(retained_variables), self.numvalues,
                                                        joint_probs=marginalized_joint_probs)

            return marginal_joint_pdf
        
    # Test function added by Cillian
    def append_variables_with_target_corr(self, num_appended_variables, target_corr, relevant_variables='all',
                                        verbose=False, num_repeats=None):
    
        # input_variables = [d for d in xrange(self.numvariables) if not d in output_variables]
    
        if relevant_variables in ('all', 'auto'):
            relevant_variables = range(len(self))
        else:
            assert len(relevant_variables) <= len(self), 'cannot be relative to more variables than I originally had'
            assert max(relevant_variables) <= len(self) - 1, 'relative to new variables...?? should not be'
    
        if target_corr == 0.0:
            raise UserWarning('you set target_corr but this is ill-defined: any independent variable(s) will do.'
                              ' Therefore you should call append_independent_variables instead and specify explicitly'
                              ' which PDFs you want to add independently.')
    
        parameter_values_before = list(self.matrix2params_incremental())
    
        pdf_new = self.copy()
        pdf_new.append_variables(num_appended_variables)
    
        assert pdf_new.numvariables == self.numvariables + num_appended_variables
    
        parameter_values_after = pdf_new.matrix2params_incremental()
    
        # this many parameters (each in [0,1]) must be optimized
        num_free_parameters = len(parameter_values_after) - len(parameter_values_before)
    
        def cost_func_target_corr(free_params, parameter_values_before):
    
            assert np.all(np.isfinite(free_params)), 'looking for bug 23142'
            # assert np.all(np.isfinite(parameter_values_before))  # todo: remove
    
            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))
    
            corr = pdf_new.pearson_correlation(relevant_variables, len(pdf_new)-1)
            #corr = pdf_new.pearson_correlation(relevant_variables, range(len(self), len(pdf_new)))
            
            #return np.power((target_corr - corr) / target_corr, 2)
            
            if np.isscalar(corr):   
                return np.power((target_corr - corr) / target_corr, 2)
            else:
                cost = 0
                for correlation in corr:
                    cost += np.power((target_corr - correlation) / target_corr, 2)
                print("cost", cost)    
                return cost
    
        self.append_optimized_variables(num_appended_variables, cost_func=cost_func_target_corr,
                                        initial_guess=np.random.random(num_free_parameters),
                                        verbose=verbose, num_repeats=num_repeats)
    
        return  # nothing, in-place
    
    def append_variables_with_target_mi(self, num_appended_variables, target_mi, relevant_variables='all',
                                        verbose=False, num_repeats=None):

        # input_variables = [d for d in xrange(self.numvariables) if not d in output_variables]

        if relevant_variables in ('all', 'auto'):
            relevant_variables = range(len(self))
        else:
            assert len(relevant_variables) <= len(self), 'cannot be relative to more variables than I originally had'
            assert max(relevant_variables) <= len(self) - 1, 'relative to new variables...?? should not be'

        if target_mi == 0.0:
            raise UserWarning('you set target_mi but this is ill-defined: any independent variable(s) will do.'
                              ' Therefore you should call append_independent_variables instead and specify explicitly'
                              ' which PDFs you want to add independently.')

        parameter_values_before = list(self.matrix2params_incremental())

        pdf_new = self.copy()
        pdf_new.append_variables(num_appended_variables)

        assert pdf_new.numvariables == self.numvariables + num_appended_variables

        parameter_values_after = pdf_new.matrix2params_incremental()

        # this many parameters (each in [0,1]) must be optimized
        num_free_parameters = len(parameter_values_after) - len(parameter_values_before)
        
        def cost_func_target_mi(free_params, parameter_values_before):

            assert np.all(np.isfinite(free_params)), 'looking for bug 23142'
            # assert np.all(np.isfinite(parameter_values_before))  # todo: remove

            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))

            mi = pdf_new.mutual_information(relevant_variables, range(len(self), len(pdf_new)))

            return np.power((target_mi - mi) / target_mi, 2)

        self.append_optimized_variables(num_appended_variables, cost_func=cost_func_target_mi,
                                        initial_guess=np.random.random(num_free_parameters),
                                        verbose=verbose, num_repeats=num_repeats)

        return  # nothing, in-place
    
    def append_variables_with_strict_target_mi(self, num_appended_variables, target_mi, relevant_variables='all',
                                        verbose=False, num_repeats=None):

        # input_variables = [d for d in xrange(self.numvariables) if not d in output_variables]
        

        # if relevant_variables in ('all', 'auto'):
        #     relevant_variables = range(len(self))
        # else:
        #     assert len(relevant_variables) <= len(self), 'cannot be relative to more variables than I originally had'
        #     assert max(relevant_variables) <= len(self) - 1, 'relative to new variables...?? should not be'
            

        # if target_mi == 0.0:
        #     raise UserWarning('you set target_mi but this is ill-defined: any independent variable(s) will do.'
        #                       ' Therefore you should call append_independent_variables instead and specify explicitly'
        #                       ' which PDFs you want to add independently.')
            
        #if relevant_variables != 'all':
        pdf_marginal = self.marginalize_distribution(relevant_variables)
        return(pdf_marginal)
        
        pdf_conditional = pdf_marginal.append_variables_with_target_mi(num_appended_variables,target_mi,relevant_variables)

        # parameter_values_before = list(self.matrix2params_incremental())

        # pdf_new = self.copy()
        # pdf_new.append_variables(num_appended_variables)

        # assert pdf_new.numvariables == self.numvariables + num_appended_variables

        # parameter_values_after = pdf_new.matrix2params_incremental()

        # # this many parameters (each in [0,1]) must be optimized
        # num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

        # def cost_func_target_mi(free_params, parameter_values_before):

        #     assert np.all(np.isfinite(free_params)), 'looking for bug 23142'
        #     # assert np.all(np.isfinite(parameter_values_before))  # todo: remove

        #     pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))

        #     mi = pdf_new.mutual_information(relevant_variables, range(len(self), len(pdf_new)))

        #     return np.power((target_mi - mi) / target_mi, 2)

        # self.append_optimized_variables(num_appended_variables, cost_func=cost_func_target_mi,
        #                                 initial_guess=np.random.random(num_free_parameters),
        #                                 verbose=verbose, num_repeats=num_repeats)

        #return  # nothing, in-place

    # helper function
    
    def appended_joint_prob_matrix(self, num_added_variables, values_so_far=[], added_joint_probabilities=None):
        if len(values_so_far) == self.numvariables:
            joint_prob_values = self.joint_probability(values_so_far)

            # submatrix must sum up to joint probability
            if added_joint_probabilities is None:
                # todo: does this add a BIAS? for a whole joint pdf it does, but not sure what I did here (think so...)
                added_joint_probabilities = np.array(np.random.random([self.numvalues]*num_added_variables),
                                                     dtype=self.type_prob)
                added_joint_probabilities /= np.sum(added_joint_probabilities)
                added_joint_probabilities *= joint_prob_values

                assert joint_prob_values <= 1.0
            else:
                np.testing.assert_almost_equal(np.sum(added_joint_probabilities), joint_prob_values)

                assert np.ndim(added_joint_probabilities) == num_added_variables
                assert len(added_joint_probabilities[0]) == self.numvalues
                assert len(added_joint_probabilities[-1]) == self.numvalues

            return list(added_joint_probabilities)
        elif len(values_so_far) < self.numvariables:
            if len(values_so_far) > 0:
                return [self.appended_joint_prob_matrix(num_added_variables,
                                                        values_so_far=list(values_so_far) + [val],
                                                        added_joint_probabilities=added_joint_probabilities)
                        for val in range(self.numvalues)]
            else:
                # same as other case but np.array converted, since the joint pdf matrix is always expected to be that
                return np.array([self.appended_joint_prob_matrix(num_added_variables,
                                                               values_so_far=list(values_so_far) + [val],
                                                               added_joint_probabilities=added_joint_probabilities)
                                 for val in range(self.numvalues)])
        else:
            raise RuntimeError('should not happen?')
    
    def append_variables(self, num_added_variables, added_joint_probabilities=None):
        assert num_added_variables > 0

        if isinstance(added_joint_probabilities, JointProbabilityMatrix) \
                or isinstance(added_joint_probabilities, NestedArrayOfProbabilities):
            added_joint_probabilities = added_joint_probabilities.joint_probabilities

        new_joint_pdf = self.appended_joint_prob_matrix(num_added_variables,
                                                        added_joint_probabilities=added_joint_probabilities)

        assert np.ndim(new_joint_pdf) == self.numvariables + num_added_variables
        if self.numvariables + num_added_variables >= 1:
            assert len(new_joint_pdf[0]) == self.numvalues
            assert len(new_joint_pdf[-1]) == self.numvalues
        np.testing.assert_almost_equal(np.sum(new_joint_pdf), 1.0)

        self.reset(self.numvariables + num_added_variables, self.numvalues, new_joint_pdf)


    
    def append_variables_using_conditional_distributions(self, cond_pdf, given_variables=None):
        """
        :param cond_pdf: dictionary of JointProbabilityMatrix objects, one for each possible set of values for the
        existing <self.numvariables> variables. If you provide a dict with one empty tuple as key then it is
        equivalent to just providing the pdf, or calling append_independent_variables with that joint pdf. If you
        provide a dict with tuples as keys which have fewer values than self.numvariables then the dict will be
        duplicated for all possible values for the remaining variables, and this function will call itself recursively
        with complete tuples.
        :type cond_pdf dict of tuple or JointProbabilityMatrix or ConditionalProbabilities
        :param given_variables: If you provide a dict with tuples as keys which have fewer values than
        self.numvariables then it is assumed that the tuples specify the values for the consecutive highest indexed
        variables (so always until variable self.numvariables-1). Unless you specify this list, whose length should
        be equal to the length of the tuple keys in pdf_conds
        :type given_variables: list of int
        """
        if type(cond_pdf) == dict:
            cond_pdf = ConditionalProbabilityMatrix(cond_pdf)  # make into conditional pdf object
        if isinstance(cond_pdf, JointProbabilityMatrix):
            cond_pdf = ConditionalProbabilityMatrix({tuple(my_values): cond_pdf for my_values in self.statespace()})
        elif isinstance(cond_pdf, ConditionalProbabilities):
            # num_conditioned_vars = len(cond_pdf.keys()[0])
            num_conditioned_vars = cond_pdf.num_given_variables()

            assert num_conditioned_vars <= len(self), 'makes no sense to condition on more variables than I have?' \
                                                      ' %s <= %s' % (num_conditioned_vars, len(self))

            if num_conditioned_vars < len(self):
                # the provided conditional pdf conditions on fewer variables (m) than I currently exist of, so appending
                # the variables is strictly speaking ill-defined. What I will do is assume that the conditional pdf
                # is conditioned on the last m variables, and simply copy the same conditional pdf for the first
                 # len(self) - m variables, which implies independence.

                pdf_conds_complete = ConditionalProbabilityMatrix()

                num_independent_vars = len(self) - num_conditioned_vars

                if __debug__:
                    # finding bug, check if all values of the dictionary are pdfs
                    for debug_pdf in iter(cond_pdf.cond_pdf.values()):
                        assert isinstance(debug_pdf, JointProbabilityMatrix), 'debug_pdf = ' + str(debug_pdf)

                statespace_per_independent_variable = [range(self.numvalues)
                                                       for _ in range(num_independent_vars)]

                if given_variables is None:
                    for indep_vals in itertools.product(*statespace_per_independent_variable):
                        pdf_conds_complete.update({(tuple(indep_vals) + tuple(key)): value
                                                   for key, value in cond_pdf.iteritems()})
                else:
                    assert len(given_variables) == len(list(cond_pdf.cond_pdf.keys())[0]), \
                        'if conditioned on ' + str(len(given_variables)) + 'then I also expect a conditional pdf ' \
                        + 'which conditions on ' + str(len(given_variables)) + ' variables.'

                    not_given_variables = np.setdiff1d(range(self.numvariables), given_variables)

                    assert len(not_given_variables) + len(given_variables) == self.numvariables

                    for indep_vals in itertools.product(*statespace_per_independent_variable):
                        pdf_conds_complete.update({tuple(apply_permutation(indep_vals + tuple(key),
                                                                           list(not_given_variables)
                                                                           + list(given_variables))): value
                                                   for key, value in iter(cond_pdf.cond_pdf.items())})

                assert len(list(pdf_conds_complete.cond_pdf.keys())[0]) == len(self)
                assert pdf_conds_complete.num_given_variables() == len(self)

                # recurse once with a complete conditional pdf, so this piece of code should not be executed again:
                self.append_variables_using_conditional_distributions(cond_pdf=pdf_conds_complete)

                return

        assert isinstance(cond_pdf, ConditionalProbabilities)

        num_added_variables = cond_pdf[(0,)*self.numvariables].numvariables

        assert cond_pdf.num_output_variables() == num_added_variables  # checking after refactoring

        # assert num_added_variables > 0, 'makes no sense to append 0 variables?'
        if num_added_variables == 0:
            return  # nothing needs to be done, 0 variables being added

        assert self.numvalues == cond_pdf[(0,)*self.numvariables].numvalues, 'added variables should have same #values'

        # see if at the end, the new joint pdf has the expected list of identifying parameter values
        # TODO: Undo this once matrix2params is implemented
        # if __debug__ and len(self) == 1:
        #     _debug_parameters_before_append = self.matrix2params()

        # note: in the loop below the sublists of parameters will be added

        statespace_per_variable = [range(self.numvalues)
                                   for _ in range(self.numvariables + num_added_variables)]

        extended_joint_probs = np.zeros([self.numvalues]*(self.numvariables + num_added_variables))

        for values in itertools.product(*statespace_per_variable):
            existing_variables_values = values[:self.numvariables]
            added_variables_values = values[self.numvariables:]

            assert len(added_variables_values) == cond_pdf[tuple(existing_variables_values)].numvariables, 'error: ' \
                'len(added_variables_values) = ' + str(len(added_variables_values)) + ', cond. numvariables = ' \
                '' + str(cond_pdf[tuple(existing_variables_values)].numvariables) + ', len(values) = ' \
                + str(len(values)) + ', existing # variables = ' + str(self.numvariables) + ', ' \
                'num_added_variables = ' + str(num_added_variables)

            prob_existing = self(existing_variables_values)
            prob_added_cond_existing = cond_pdf[tuple(existing_variables_values)](added_variables_values)

            assert 0.0 <= prob_existing <= 1.0, 'prob not normalized'
            assert 0.0 <= prob_added_cond_existing <= 1.0, 'prob not normalized'

            extended_joint_probs[tuple(values)] = prob_existing * prob_added_cond_existing

            # TODO: Uncomment once matrix2params is done
            # if __debug__ and len(self) == 1:
            #     _debug_parameters_before_append.append(cond_pdf[tuple(existing_variables_values)].matrix2params_incremental)

        if __debug__ and np.random.randint(10) == 0:
            np.testing.assert_almost_equal(np.sum(extended_joint_probs), 1.0)

        self.reset(self.numvariables + num_added_variables, self.numvalues, extended_joint_probs)

        # if __debug__ and len(self) == 1:
        #     # of course this test depends on the implementation of matrix2params_incremental, currently it should
        #     # work
        #     np.testing.assert_array_almost_equal(self.scalars_up_to_level(_debug_parameters_before_append),
        #                                          self.matrix2params_incremental(return_flattened=True))

    def conditional_probability_distributions(self, given_variables, object_variables='auto', nprocs=1):
        """

        :param given_variables:
        :return: dict of JointProbabilityMatrix, keys are all possible values for given_variables
        :rtype: dict of JointProbabilityMatrix
        """
        if len(given_variables) == self.numvariables:  # 'no variables left after conditioning'
            warnings.warn('conditional_probability_distributions: no variables left after conditioning')

            lists_of_possible_given_values = [range(self.numvalues) for variable in range(len(given_variables))]

            dic = {tuple(given_values): JointProbabilityMatrix(0, self.numvalues)
                   for given_values in itertools.product(*lists_of_possible_given_values)}

            return ConditionalProbabilityMatrix(dic)
        else:
            if object_variables in ('auto', 'all'):
                object_variables = list(np.setdiff1d(range(len(self)), given_variables))
            else:
                object_variables = list(object_variables)
                ignored_variables = list(np.setdiff1d(range(len(self)), list(given_variables) + object_variables))

                pdf = self.copy()
                pdf.reorder_variables(list(given_variables) + object_variables + ignored_variables)
                # note: I do this extra step because reorder_variables does not support specifying fewer variables
                # than len(self), but once it does then it can be combined into a single reorder_variables call.
                # remove ignored_variables
                pdf = pdf.marginalize_distribution(range(len(given_variables + object_variables)))

                return pdf.conditional_probability_distributions(range(len(given_variables)))

            lists_of_possible_given_values = [range(self.numvalues) for variable in range(len(given_variables))]

            if nprocs == 1:
                dic = {tuple(given_values): self.conditional_probability_distribution(given_variables=given_variables,
                                                                                      given_values=given_values)
                       for given_values in itertools.product(*lists_of_possible_given_values)}
            else:
                def worker_cpd(given_values):  # returns a 2-tuple
                    return (tuple(given_values),
                            self.conditional_probability_distribution(given_variables=given_variables,
                                                                      given_values=given_values))

                pool = mp.Pool(nprocs)
                dic = pool.map(worker_cpd, itertools.product(*lists_of_possible_given_values))
                dic = dict(dic)  # make a dictionary out of it
                pool.close()
                pool.terminate()

            return ConditionalProbabilityMatrix(dic)
    
    def conditional_probability_distribution(self, given_variables, given_values):
        """

        :param given_variables: list of integers
        :param given_values: list of integers
        :rtype: JointProbabilityMatrix
        """
        assert len(given_values) == len(given_variables)
        assert len(given_variables) < self.numvariables, 'no variables left after conditioning'

        lists_of_possible_states_per_variable = [range(self.numvalues) for variable in range(self.numvariables)]

        # overwrite the 'state spaces' for the specified variables, to the specified state spaces
        for gix in range(len(given_variables)):
            assert np.isscalar(given_values[gix]), 'assuming specific value, not list of possibilities'

            lists_of_possible_states_per_variable[given_variables[gix]] = \
                [given_values[gix]] if np.isscalar(given_values[gix]) else given_values[gix]

        conditioned_variables = [varix for varix in range(self.numvariables) if varix not in given_variables]

        conditional_probs = np.zeros([self.numvalues]*len(conditioned_variables))

        assert len(conditional_probs) > 0, 'you want me to make a conditional pdf of 0 variables?'

        assert len(given_variables) + len(conditioned_variables) == self.numvariables

        for values in itertools.product(*lists_of_possible_states_per_variable):
            values_conditioned_vars = [values[varid] for varid in conditioned_variables]

            assert conditional_probs[tuple(values_conditioned_vars)] == 0.0, 'right?'

            # note: here self.joint_probability(values) == Pr(conditioned_values | given_values) because the
            # given_variables == given_values constraint is imposed on the set
            # itertools.product(*lists_of_possible_states_per_variable); it is only not yet normalized
            conditional_probs[tuple(values_conditioned_vars)] += self.joint_probability(values)

        summed_prob_mass = np.sum(conditional_probs)

        # testing here if the summed prob. mass equals the marginal prob of the given variable values
        if __debug__:
            # todo: can make this test be run probabilistically, like 10% chance or so, pretty expensive?
            if np.all(map(np.isscalar, given_values)):
                pdf_marginal = self.marginalize_distribution(given_variables)

                prob_given_values = pdf_marginal(given_values)

                np.testing.assert_almost_equal(prob_given_values, summed_prob_mass)

        assert np.isscalar(summed_prob_mass), 'sum of probability mass should be a scalar, not a list or so: ' \
                                              + str(summed_prob_mass)
        assert np.isfinite(summed_prob_mass)

        assert summed_prob_mass >= 0.0, 'probability mass cannot be negative'

        if summed_prob_mass > 0.0:
            conditional_probs /= summed_prob_mass
        else:
            # note: apparently the probability of the given condition is zero, so it does not really matter
            # what I substitute for the probability mass here. I will add some probability mass so that I can
            # normalize it.

            # # I think this warning can be removed at some point...
            # warnings.warn('conditional_probability_distribution: summed_prob_mass == 0.0 (can be ignored)')

            # are all values zero?
            try:
                np.testing.assert_almost_equal(np.min(conditional_probs), 0.0)
            except ValueError as e:
                print('debug: conditional_probs =', conditional_probs)
                print('debug: min(conditional_probs) =', min(conditional_probs))

                raise ValueError(e)
                
            if __debug__:
                np.testing.assert_almost_equal(np.max(conditional_probs), 0.0)

            conditional_probs *= 0
            conditional_probs += 1.0  # create some fake mass, making it a uniform distribution

            conditional_probs /= np.sum(conditional_probs)

        conditional_joint_pdf = JointProbabilityMatrix(len(conditioned_variables), self.numvalues,
                                                       joint_probs=conditional_probs)

        return conditional_joint_pdf
    
    #Cillian added
    def corr_probs(self, variables=None):

        if variables is None:
    
            if __debug__ and np.random.random() < 0.1:
                np.testing.assert_almost_equal(self.joint_probabilities.sum(), 1.0)
    
            probs = self.joint_probabilities.flatten()
            probs = np.select([probs != 0], [probs], default=1)
    
            return probs
        else:
            assert hasattr(variables, '__iter__')
    
            if len(variables) == 0:  # hard-coded this because otherwise I have to support empty pdfs (len() = 0)
                return 0.0
    
            marginal_pdf = self.marginalize_distribution(retained_variables=variables)
    
            return marginal_pdf.corr_probs(None)
                
    def entropy(self, variables=None, base=2):

        if variables is None:

            if __debug__ and np.random.random() < 0.1:
                np.testing.assert_almost_equal(self.joint_probabilities.sum(), 1.0)

            probs = self.joint_probabilities.flatten()
            probs = np.select([probs != 0], [probs], default=1)

            if base == 2:
                # joint_entropy = np.sum(map(log2term, self.joint_probabilities.flatiter()))
                log_terms = probs * np.log2(probs)
                # log_terms = np.select([np.isfinite(log_terms)], [log_terms])
                joint_entropy = -np.sum(log_terms)
            elif base == np.e:
                # joint_entropy = np.sum(map(log2term, self.joint_probabilities.flatiter()))
                log_terms = probs * np.log(probs)
                # log_terms = np.select([np.isfinite(log_terms)], [log_terms])
                joint_entropy = -np.sum(log_terms)
            else:
                # joint_entropy = np.sum(map(log2term, self.joint_probabilities.flatiter()))
                log_terms = probs * (np.log(probs) / np.log(base))
                # log_terms = np.select([np.isfinite(log_terms)], [log_terms])
                joint_entropy = -np.sum(log_terms)

            assert joint_entropy >= 0.0

            return joint_entropy
        else:
            assert hasattr(variables, '__iter__')

            if len(variables) == 0:  # hard-coded this because otherwise I have to support empty pdfs (len() = 0)
                return 0.0

            marginal_pdf = self.marginalize_distribution(retained_variables=variables)

            return marginal_pdf.entropy(base=base)
    #   
    def conditional_entropy(self, variables, given_variables=None):
        assert hasattr(variables, '__iter__'), 'variables1 = ' + str(variables)
        assert hasattr(given_variables, '__iter__') or given_variables is None, 'variables2 = ' + str(given_variables)

        assert max(variables) < self.numvariables, 'variables are 0-based indices, so <= N - 1: variables=' \
                                                   + str(variables) + ' (N=' + str(self.numvariables) + ')'

        if given_variables is None:
            # automatically set the given_variables to be the complement of 'variables', so all remaining variables

            given_variables = [varix for varix in range(self.numvariables) if not varix in variables]

            assert len(set(variables)) + len(given_variables) == self.numvariables, 'variables=' + str(variables) \
                                                                                    + ', given_variables=' \
                                                                                    + str(given_variables)

        # H(Y) + H(X|Y) == H(X,Y)
        condent = self.entropy(list(set(list(variables) + list(given_variables)))) - self.entropy(given_variables)

        assert np.isscalar(condent)
        assert np.isfinite(condent)

        assert condent >= 0.0, 'conditional entropy should be non-negative'

        return condent
    
    def mutual_information_labels(self, labels1, labels2):
        variables1 = self.get_variables_with_label_in(labels1)
        variables2 = self.get_variables_with_label_in(labels2)

        return self.mutual_information(variables1, variables2)
    
    def conditional_mutual_informations(self, variables1, variables2, given_variables='all', nprocs=1):

        if given_variables == 'all':
            # automatically set the given_variables to be the complement of 'variables', so all remaining variables

            given_variables = [varix for varix in range(self.numvariables)
                               if not varix in list(variables1) + list(variables2)]

        # pZ = self.marginalize_distribution(given_variables)

        cond_pXY_given_z = self.conditional_probability_distributions(given_variables, nprocs=nprocs)

        varXnew = list(variables1)
        varYnew = list(variables2)

        # compute new indices of X and Y variables in conditioned pdfs, cond_pXY_given_z
        for zi in sorted(given_variables, reverse=True):
            for ix in range(len(varXnew)):
                assert varXnew[ix] != zi
                if varXnew[ix] > zi:
                    varXnew[ix] -= 1
            for ix in range(len(varYnew)):
                assert varYnew[ix] != zi
                if varYnew[ix] > zi:
                    varYnew[ix] -= 1

        mi_given_z = dict()

        # todo: also parallellize over nprocs cpus?
        for z in self.statespace(len(given_variables)):
            mi_given_z[z] = cond_pXY_given_z[z].mutual_information(varXnew, varYnew)

        return mi_given_z
    
    def conditional_mutual_information(self, variables1, variables2, given_variables='all'):

        if given_variables == 'all':
            # automatically set the given_variables to be the complement of 'variables', so all remaining variables

            given_variables = [varix for varix in range(self.numvariables)
                               if not varix in list(variables1) + list(variables2)]

        condmis = self.conditional_mutual_informations(variables1, variables2, given_variables)
        p3 = self.marginalize_distribution(given_variables)

        return sum([p3(z) * condmis[z] for z in p3.statespace()])
    
    # Test function added by Cillian
    def pearson_correlation(self, variables1, variables2):
        assert hasattr(variables1, '__iter__'), 'variables1 = ' + str(variables1)
        assert hasattr(variables2, '__iter__'), 'variables2 = ' + str(variables2)
    
        if len(variables1) == 0 or len(variables2) == 0:
            corr = 0  # trivial case, no computation needed
        # elif len(variables1) == len(variables2):
        #     assert max(variables1) < len(self), 'variables1 = ' + str(variables1) + ', len(self) = ' + str(len(self))
        #     assert max(variables2) < len(self), 'variables2 = ' + str(variables2) + ', len(self) = ' + str(len(self))
    
        #     if np.equal(sorted(variables1), sorted(variables2)).all():
        #         mi = self.entropy(variables1, base=base)  # more efficient, shortcut
        #     else:
        #         # this one should be equal to the outer else-clause (below), i.e., the generic case
        #         mi = self.entropy(variables1, base=base) + self.entropy(variables2, base=base) \
        #              - self.entropy(list(set(list(variables1) + list(variables2))), base=base)
        else:
            assert max(variables1) < len(self), 'variables1 = ' + str(variables1) + ', len(self) = ' + str(len(self))
            assert max(variables2) < len(self), 'variables2 = ' + str(variables2) + ', len(self) = ' + str(len(self))
    
            # mi = self.entropy(variables1, base=base) + self.entropy(variables2, base=base) \
            #      - self.entropy(list(set(list(variables1) + list(variables2))), base=base)
                 
            #corr = np.corrcoef(variables1,variables2)[0]
            #print("old attempt",corr)
            
            jpd = self.joint_probabilities.joint_probabilities
            #print("jpd shape = ",jpd.shape)
            #print("variables1 = ",variables1)
            #print("variables2 = ",variables2[0])
            
            axis_sum = range(len(self))
            axis_sum = list(axis_sum)
            
            if len(variables1) == 1:
            
                axis_sum.remove(variables1[0])
                axis_sum.remove(variables2[0])
                axis_sum = tuple(axis_sum)
                #print("axis_sum = ",axis_sum)
                jpd = jpd.sum(axis = axis_sum)
                
                a = jpd[1,1]
                p = jpd[1,0] + jpd[1,1]
                q = jpd[0,1] + jpd[1,1]
                
                #print("ad-bc",jpd[1,1]*jpd[0,0] - jpd[1,0]*jpd[0,1] )
                #print("denominitor = ",  (p*(1-p)**0.5)*(q*(1-q)**0.5) )
                
                corr = (a - p*q)/(( (p*(1-p))**0.5) *( (q*(1-q))**0.5) )
                assert np.isscalar(corr)
                assert np.isfinite(corr)
                
                assert corr >= -1, 'pearson correlation should be greater than -1: ' + str(corr)
                
                assert corr <= 1, 'pearson correlation should be less than 1: ' + str(corr)
            
            elif len(variables1) > 1:
                #print("Dont know how to handle multiple dependent variables!")
                corr = np.empty((len(variables1)))
                
                for _,dependent_variable in enumerate(variables1):
                    jpd = self.joint_probabilities.joint_probabilities
                    axis_sum = range(len(self))
                    axis_sum = list(axis_sum)
                    
                    axis_sum.remove(dependent_variable)
                    axis_sum.remove(variables2[0])
                    axis_sum = tuple(axis_sum)
                    #print("axis_sum = ",axis_sum)
                    jpd = jpd.sum(axis = axis_sum)
                    
                    a = jpd[1,1]
                    p = jpd[1,0] + jpd[1,1]
                    q = jpd[0,1] + jpd[1,1]
                    
                    #print("ad-bc",jpd[1,1]*jpd[0,0] - jpd[1,0]*jpd[0,1] )
                    #print("denominitor = ",  (p*(1-p)**0.5)*(q*(1-q)**0.5) )
                    
                    cors = (a - p*q)/(( (p*(1-p))**0.5) *( (q*(1-q))**0.5) )
                    
                    assert np.isscalar(cors)
                    assert np.isfinite(cors)
                    
                    assert cors >= -1, 'pearson correlation should be greater than -1: ' + str(cors)
                    
                    assert cors <= 1, 'pearson correlation should be less than 1: ' + str(cors)
                    
                    
                    corr[_] = cors
                
                
                #print("ok..")
                #print(corr)
                #raise NotImplementedError('Dont know how to handle multiple dependent variables...')
                
            else:
                raise NotImplementedError('There should be at least one dependent variable...')
                
            
            #p_A = jpd.sum(axis = (0))
            #p_B = jpd.sum(axis = (1))
            #print("p_A",p_A,"p_B",p_B)
            #corr = (jpd[1,1] - p_A[1]*p_B[1]) / ((p_A[1] - p_A[1]*p_A[1])**(0.5))*((p_B[1] - p_B[1]*p_B[1])**(0.5)) 
            #print("corr=",corr)
            #corr = round(corr,4) # will rounding help?
            
            # probs1 = self.corr_probs(variables1)
            # print("probs 1: ",probs1 )
            # probs2 = self.corr_probs(variables2)
            # print("probs 2: ",probs2 )
            # corr = np.corrcoef(probs1,probs2)[0][1]
            # print("new attempt: Var has correlation of ",corr)
            # print("Corr matrix:", np.corrcoef(probs1,probs2))
            # print("Corr matrix:", np.corrcoef(probs1,probs2, rowvar=False))
    
        #print("Scalar?",np.isscalar(corr))
        #assert np.isscalar(corr)
        #assert np.isfinite(corr)
    
        # # due to floating-point operations it might VERY sometimes happen that I get something like -4.4408920985e-16
        # # here, so to prevent this case from firing an assert I clip this to 0:
        # if -0.000001 < mi < 0.0:
        #     mi = 0.0
    
        # assert corr >= -1, 'pearson correlation should be greater than -1: ' + str(corr)
        
        # assert corr <= 1, 'pearson correlation should be less than 1: ' + str(corr)
    
        return corr
    
    def mutual_information(self, variables1, variables2, base=2):
        assert hasattr(variables1, '__iter__'), 'variables1 = ' + str(variables1)
        assert hasattr(variables2, '__iter__'), 'variables2 = ' + str(variables2)

        if len(variables1) == 0 or len(variables2) == 0:
            mi = 0  # trivial case, no computation needed
        elif len(variables1) == len(variables2):
            assert max(variables1) < len(self), 'variables1 = ' + str(variables1) + ', len(self) = ' + str(len(self))
            assert max(variables2) < len(self), 'variables2 = ' + str(variables2) + ', len(self) = ' + str(len(self))

            if np.equal(sorted(variables1), sorted(variables2)).all():
                mi = self.entropy(variables1, base=base)  # more efficient, shortcut
            else:
                # this one should be equal to the outer else-clause (below), i.e., the generic case
                mi = self.entropy(variables1, base=base) + self.entropy(variables2, base=base) \
                     - self.entropy(list(set(list(variables1) + list(variables2))), base=base)
        else:
            assert max(variables1) < len(self), 'variables1 = ' + str(variables1) + ', len(self) = ' + str(len(self))
            assert max(variables2) < len(self), 'variables2 = ' + str(variables2) + ', len(self) = ' + str(len(self))

            mi = self.entropy(variables1, base=base) + self.entropy(variables2, base=base) \
                 - self.entropy(list(set(list(variables1) + list(variables2))), base=base)

        assert np.isscalar(mi)
        assert np.isfinite(mi)

        # due to floating-point operations it might VERY sometimes happen that I get something like -4.4408920985e-16
        # here, so to prevent this case from firing an assert I clip this to 0:
        if -0.000001 < mi < 0.0:
            mi = 0.0

        assert mi >= 0, 'mutual information should be non-negative: ' + str(mi)

        return mi
        
    def mutual_informations(self, variables1, variables2, base=2):
        assert hasattr(variables1, '__iter__'), 'variables1 = ' + str(variables1)
        assert hasattr(variables2, '__iter__'), 'variables2 = ' + str(variables2)

        if len(variables1) == 0 or len(variables2) == 0:
            mi = 0  # trivial case, no computation needed
        elif len(variables1) == len(variables2):
            assert max(variables1) < len(self), 'variables1 = ' + str(variables1) + ', len(self) = ' + str(len(self))
            assert max(variables2) < len(self), 'variables2 = ' + str(variables2) + ', len(self) = ' + str(len(self))

            if np.equal(sorted(variables1), sorted(variables2)).all():
                mi = self.entropy(variables1, base=base)  # more efficient, shortcut
            else:
                # this one should be equal to the outer else-clause (below), i.e., the generic case
                mi = self.entropy(variables1, base=base) + self.entropy(variables2, base=base) \
                     - self.entropy(list(set(list(variables1) + list(variables2))), base=base)
        else:
            assert max(variables1) < len(self), 'variables1 = ' + str(variables1) + ', len(self) = ' + str(len(self))
            assert max(variables2) < len(self), 'variables2 = ' + str(variables2) + ', len(self) = ' + str(len(self))

            mi = self.entropy(variables1, base=base) + self.entropy(variables2, base=base) \
                 - self.entropy(list(set(list(variables1) + list(variables2))), base=base)

        assert np.isscalar(mi)
        assert np.isfinite(mi)

        # due to floating-point operations it might VERY sometimes happen that I get something like -4.4408920985e-16
        # here, so to prevent this case from firing an assert I clip this to 0:
        if -0.000001 < mi < 0.0:
            mi = 0.0

        assert mi >= 0, 'mutual information should be non-negative: ' + str(mi)

        return mi
    
    def loglikelihood(self, repeated_measurements, ignore_zeros=True):
        if not ignore_zeros:
            return np.sum(np.log(map(self, repeated_measurements)))
        else:
            return np.sum(np.log([p for p in map(self, repeated_measurements) if p > 0.]))
    
    def estimate_from_data(self, repeated_measurements, numvalues='auto', discrete=True, method='empirical'):
        """
        From a list of co-occurring values (one per variable) create a joint probability distribution by simply
        counting. For instance, the list [[0,0], [0,1], [1,0], [1,1]] would result in a uniform distribution of two
        binary variables.
        :param repeated_measurements: list of lists, where each sublist is of equal length (= number of variables)
        :type repeated_measurements: list of list
        :param discrete: do not change
        :param method: do not change
        """
        if not discrete and method == 'empirical':
            method = 'equiprobable'  # 'empirical' makes no sense for continuous data, so pick most sensible alternative

        assert discrete and method == 'empirical' or not discrete and method in ('equiprobable', 'equiprob'), \
            'method/discreteness combination not supported'

        assert len(repeated_measurements) > 0, 'no data given'

        numvars = len(repeated_measurements[0])

        assert numvars <= len(repeated_measurements), 'fewer measurements than variables, probably need to transpose' \
                                                      ' repeated_measurements.'

        if discrete and method == 'empirical':
            all_unique_values = list(set(np.array(repeated_measurements).flatten()))

            # todo: store the unique values as a member field so that later algorithms know what original values
            # corresponded to the values 0, 1, 2, ... and for estimating MI based on samples rather than 'true' dist.

            if numvalues == 'auto':
                numvals = len(all_unique_values)
            else:
                numvals = int(numvalues)

                for v in range(2**31):
                    if len(all_unique_values) < numvals:
                        if not v in all_unique_values:
                            all_unique_values.append(v)  # add bogus values
                    else:
                        break

            dict_val_to_index = {all_unique_values[valix]: valix for valix in range(numvals)}

            new_joint_probs = np.zeros([numvals]*numvars)

            # todo: when you generalize self.numvalues to an array then here also set the array instead of int

            for values in repeated_measurements:
                value_indices = tuple((dict_val_to_index[val] for val in values))

                try:
                    new_joint_probs[value_indices] += 1.0 / len(repeated_measurements)
                except IndexError as e:
                    print('error: value_indices =', value_indices)
                    print('error: type(value_indices) =', type(value_indices))

                    raise IndexError(e)

            self.reset(numvars, numvals, joint_prob_matrix=new_joint_probs)
        elif not discrete and method in ('equiprobable', 'equiprob'):
            disc_timeseries = self.discretize_data(repeated_measurements, numvalues, return_also_fitness_curve=False)

            assert np.shape(repeated_measurements) == np.shape(disc_timeseries)

            self.estimate_from_data(disc_timeseries, numvalues=numvalues, discrete=True, method='empirical')
        else:
            raise NotImplementedError('unknown combination of discrete and method.')
    
    def discretize_data(repeated_measurements, numvalues='auto', return_also_fitness_curve=False, maxnumvalues=20,
                    stopafterdeclines=5, method='equiprobable'):

        pdf = JointProbabilityMatrix(2, 2)  # pre-alloc, values passed are irrelevant

        # assert numvalues == 'auto', 'todo'

        timeseries = np.transpose(repeated_measurements)

        fitness = []  # tuples of (numvals, fitness)

        # print('debug: len(rep) = %s' % len(repeated_measurements)
        # print('debug: len(ts) = %s' % len(timeseries)
        # print('debug: ub =', xrange(2, min(max(int(np.power(len(repeated_measurements), 1.0/2.0)), 5), 100))

        if numvalues == 'auto':
            possible_numvals = range(2, min(max(int(np.power(len(repeated_measurements), 1.0/2.0)), 5), maxnumvalues))
        else:
            possible_numvals = [numvalues]  # try only one

        for numvals in possible_numvals:
            if method in ('equiprobable', 'adaptive'):
                bounds = [[np.percentile(ts, p) for p in np.linspace(0, 100., numvals + 1)[:-1]] for ts in timeseries]
            elif method in ('equidistant', 'fixed'):
                bounds = [[np.min(ts) + p * np.max(ts) for p in np.linspace(0., 1., numvals + 1)[:-1]] for ts in timeseries]
            else:
                raise NotImplementedError('unknown method: %s' % method)

            # NOTE: I suspect it is not entirely correct to discretize the entire timeseries and then compute
            # likelihood of right side given left...

            disc_timeseries = [[np.sum(np.less_equal(bounds[vix], val)) - 1 for val in ts]
                               for vix, ts in enumerate(timeseries)]

            assert np.min(disc_timeseries) >= 0, 'discretized values should be non-negative'
            assert np.max(disc_timeseries) < numvals, \
                'discretized values should be max. numvals-1, max=%s, numvals=%s' % (np.max(disc_timeseries), numvals)

            # estimate pdf on FIRST HALF of the data to predict the SECOND HALF (two-way cross-validation)
            pdf.estimate_from_data(np.transpose(disc_timeseries)[:int(len(repeated_measurements)/2)], numvalues=numvals)
            fitness_nv = pdf.loglikelihood(np.transpose(disc_timeseries)[int(len(repeated_measurements)/2):])
            # correction for number of variables?
            # this is the expected likelihood for independent variables if the pdf fits perfect
            fitness_nv /= np.log(1.0/np.power(numvals, len(pdf)))

            # estimate pdf on SECOND HALF of the data to predict the FIRST HALF
            pdf.estimate_from_data(np.transpose(disc_timeseries)[int(len(repeated_measurements) / 2):], numvalues=numvals)
            fitness_nv_r = pdf.loglikelihood(np.transpose(disc_timeseries)[:int(len(repeated_measurements) / 2)])
            # correction for number of variables?
            # this is the expected likelihood for independent variables if the pdf fits perfect
            fitness_nv_r /= np.log(1.0 / np.power(numvals, len(pdf)))

            fitness_nv = fitness_nv + fitness_nv_r  # combined (avg) fitness

            fitness.append((numvals, np.transpose(disc_timeseries), fitness_nv))
            # NOTE: very inefficient to store all potential disc_timeseries
            # todo: fix that

            assert np.max(disc_timeseries) <= numvals, \
                'there should be %s values but there are %s' % (numvals, np.max(disc_timeseries))

            print('debug: processed %s, fitness=%s' % (numvals, fitness_nv))
            if fitness_nv <= 0.0:
                print('debug: disc. timeseries:', np.transpose(disc_timeseries)[:10])
                print('debug: repeated_measurements:', np.transpose(repeated_measurements)[:10])
                print('debug: bounds:', np.transpose(bounds)[:10])

            if True:
                fitness_values = map(lambda x: x[-1], fitness)

                if len(fitness) > 7:
                    if fitness_nv < max(fitness_values) * 0.5:
                        print('debug: not going to improve, will break the for-loop')
                        break

                if len(fitness) > stopafterdeclines:
                    if list(fitness_values[-stopafterdeclines:]) == sorted(fitness_values[-stopafterdeclines:]):
                        print('debug: fitness declined %s times in a row, will stop' % stopafterdeclines)
                        break

        max_ix = np.argmax(map(lambda x: x[-1], fitness), axis=0)

        if not return_also_fitness_curve:
            return fitness[max_ix][1]
        else:
            return fitness[max_ix][1], map(lambda x: (x[0], x[-1]), fitness)
    
    def scalars_up_to_level(self, list_of_lists, max_level=None):
        """
        Helper function. E.g. scalars_up_to_level([1,[2,3],[[4]]]) == [1], and
        scalars_up_to_level([1,[2,3],[[4]]], max_level=2) == [1,2,3]. Will be sorted on level, with highest level
        scalars first.

        Note: this function is not very efficiently implemented I think, but my concern now is that it works at all.

        :type list_of_lists: list
        :type max_level: int
        :rtype: list
        """
        # scalars = [v for v in list_of_lists if np.isscalar(v)]
        #
        # if max_level > 1 or (max_level is None and len(list_of_lists) > 0):
        #     for sublist in [v for v in list_of_lists if not np.isscalar(v)]:
        #         scalars.extend(self.scalars_up_to_level(sublist,
        #                                                 max_level=max_level-1 if not max_level is None else None))

        scalars = []

        if __debug__:
            debug_max_depth_set = (max_level is None)

        if max_level is None:
            max_level = maximum_depth(list_of_lists)

        for at_level in range(1, max_level + 1):
            scalars_at_level = self.scalars_at_level(list_of_lists, at_level=at_level)

            scalars.extend(scalars_at_level)

        if __debug__:
            if debug_max_depth_set:
                assert len(scalars) == len(flatten(list_of_lists)), 'all scalars should be present, and not duplicate' \
                                                                    '. len(scalars) = ' + str(len(scalars)) \
                                                                    + ', len(flatten(list_of_lists)) = ' \
                                                                    + str(len(flatten(list_of_lists)))

        return scalars
    
    def scalars_at_level(self, list_of_lists, at_level=1):
        """
        Helper function. E.g. scalars_up_to_level([1,[2,3],[[4]]]) == [1], and
        scalars_up_to_level([1,[2,3],[[4]]], max_level=2) == [1,2,3]. Will be sorted on level, with highest level
        scalars first.
        :type list_of_lists: list
        :type max_level: int
        :rtype: list
        """

        if at_level == 1:
            scalars = [v for v in list_of_lists if np.isscalar(v)]

            return scalars
        elif at_level == 0:
            warnings.warn('level 0 does not exist, I start counting from at_level=1, will return [].')

            return []
        else:
            scalars = []

            for sublist in [v for v in list_of_lists if not np.isscalar(v)]:
                scalars.extend(self.scalars_at_level(sublist, at_level=at_level-1))

            assert np.ndim(scalars) == 1

            return scalars
        
    # Cillian Added
    # helper function
    def generate_nudge(self, epsilon, shape):
        nudge_vec = np.random.dirichlet([1] * np.product(shape))
        nudge_vec -= 1. / np.product(shape)
        norm = np.linalg.norm(nudge_vec)
        nudge_vec = np.reshape(nudge_vec / norm * epsilon, newshape=shape)
    
        return nudge_vec
    
    def nudge_single(self, perturbed=0, eps_norm=0.01, method='invariant'):
        """
        In a bivariate distribution p(X,Y), nudge a second variable Y's marginal probabilities
        without affecting the first (X). It does this by changing the conditional probabilities p(y|x).
    
        Note: at the moment I have written this for the special case of len(self)==2, so an X and a Y. Should
        generalize at some point.
        :param perturbed: which variable index to nudge. Must be an index in the range 0..numvariables.
        :param eps_norm: norm of zero-sum vector to be added to Y's marginal probability vector.
        :type ntrials: int
        :returns: nudge vector applied.
        :param method: 'invariant' means that a single nudge vector is generated and applied to all conditional
        distributions p(Y|X=x) for all x
        :rtype: np.array
        """
    
        assert method == 'invariant', 'no other method implemented yet'
    
        # in the notation: X = variable(s) which are not nudged AND not effected by the nudge (causal predecessors);
        # Y = perturbed variable(s). Z = downstream variables potentially influenced by the nudge (not nudged themselves)
    
        # note: code below assumes perfect ordering (X, Y, Z) and contiguous ranges of indices (xix+yix+zix=range(n))
        xix = list(range(perturbed))
        yix = list(range(perturbed, perturbed + 1))
        zix = list(range(perturbed + 1, self.numvariables))
    
        pdfXYZ = self  # shorthand
        pdfXY = self.marginalize_distribution(xix + yix)  # shorthand, readable
    
        # assert len(zix) == 0, 'currently assumed by the code, but not hard to implement (store cond. pdf and then add back)'
        assert len(yix) > 0, 'set of variables to nudge should not be empty'
    
        pX = pdfXY.marginalize_distribution(xix)
        pY_X = pdfXY.conditional_probability_distributions(xix, yix)
        pY = pdfXY.marginalize_distribution(yix)
    
        # remove Z but later add it/them back by means of their conditional pdf
        pZ_XY = pdfXYZ.conditional_probability_distributions(xix + yix, zix)
    
        nY = len(list(pY.statespace()))  # shorthand
    
        # print 'debug: epsilon: %s -- norm: %s  (after clipping 1)' % (epsilon, np.linalg.norm(epsilon))
    
        # make pY_X in list format, condprobs[yi][xi] is p(yi|xi)
        condprobs = np.array([[pY_X[xi](yi) for xi in pX.statespace()] for yi in pY.statespace()], dtype=np.float64)
    
        # np.testing.assert_array_almost_equal(np.sum(condprobs, axis=0), np.ones(nY))
    
        # note: in total, sum_xi condprobs[yi][xi] must change by amount epsilon[yi], but we
        # have to divide this change into |Y| subterms which sum to epsilon[yi]...
    
        pXprobs = np.array(pX.joint_probabilities.joint_probabilities, dtype=np.float64)
    
        # np.testing.assert_array_almost_equal(
        #     [np.sum(condprobs[yix] * pXprobs) for yix, yi in enumerate(pY.statespace())],
        #     pY.joint_probabilities.joint_probabilities)
        # np.testing.assert_array_almost_equal(np.sum(pXprobs * condprobs, axis=1),
        #                                      pY.joint_probabilities.joint_probabilities)
    
        # across all conditional distributions p(Y|X=x) these are the min. and max. probabilities for Y, so the
        # nudge vector (which will be added to all vectors p(Y|X=x) for all x) cannot exceed these bounds since
        # otherwise there will be probabilities out of the range [0,1]
        min_cond_probs_Y = np.min(condprobs, axis=1)
        max_cond_probs_Y = np.max(condprobs, axis=1)
    
        tol_sum_eps = 1e-15
        ntrials_clip = 30
        tol_norm_rel = 0.1  # relative error allowed for the norm of the epsilon vector (0.1=10% error)
        ntrials_norm = 20
    
        for j in range(ntrials_norm):
            # generate a nudge vector
            epsilon = pY.generate_nudge(eps_norm, np.shape(pY.joint_probabilities.joint_probabilities))
    
            # print 'debug: epsilon: %s -- norm: %s' % (epsilon, np.linalg.norm(epsilon))
    
            # clip the nudge vector to let all probabilities stay within bounds
            # epsilon = np.max([epsilon, -pY.joint_probabilities.joint_probabilities], axis=0)
            # epsilon = np.min([epsilon, 1.0 - pY.joint_probabilities.joint_probabilities], axis=0)
            epsilon = np.array(epsilon, dtype=np.float64)  # try to reduce roundoff errors below
    
            # clip the nudge vector to stay within probabilities [0,1]
            # TODO: generate directly a nudge vector within these bounds? in the helper function?
            for i in range(ntrials_clip):  # try to clip the nudge vector within a number of trials
                epsilon = np.max([epsilon, -min_cond_probs_Y], axis=0)
                epsilon = np.min([epsilon, 1.0 - max_cond_probs_Y], axis=0)
    
                if np.abs(np.sum(epsilon)) < tol_sum_eps:
                    # print 'debug: took %s trials to find a good epsilon after clipping' % (i + 1)
                    break
                else:
                    epsilon -= np.mean(epsilon)
    
            if not np.abs(np.sum(epsilon)) < tol_sum_eps:
                # print 'debug: did not manage to make a nudge vector sum to zero after clipping'
                pass
            elif np.linalg.norm(epsilon) < eps_norm * (1.0 - tol_norm_rel):
                # print 'debug: did not manage to keep the norm within tolerance (%s)' % np.linalg.norm(epsilon)
                pass
            else:
                # print 'debug: epsilon: %s -- norm: %s -- sum %s (trial %s)' % (epsilon, np.linalg.norm(epsilon), np.sum(epsilon), j+1)
                break  # successfully found an epsilon vector that matches the desired norm and sums to approx. zero
    
        if not np.abs(np.sum(epsilon)) < tol_sum_eps:
            raise UserWarning('debug: did not manage to make a nudge vector sum to zero after clipping (%s)' % np.sum(epsilon))
        elif np.linalg.norm(epsilon) < eps_norm * (1.0 - tol_norm_rel):
            raise UserWarning('debug: did not manage to keep the norm within tolerance (%s)' % np.linalg.norm(epsilon))
    
        nudged_pdf = pX.copy()  # first add the (unchanged) X
    
        new_pY_x = lambda x: pdfXY.conditional_probability_distribution(xix, x).joint_probabilities.joint_probabilities + epsilon
    
        nudged_pdf.append_variables_using_conditional_distributions({x: JointProbabilityMatrix(len(yix), nY, new_pY_x(x))
                                                                     for x in pX.statespace()}, xix)
    
        # add back the Z through the conditional pdf (so p(Z) may now be changed if p(Z|Y) != p(Z))
        nudged_pdf.append_variables_using_conditional_distributions(pZ_XY)
    
        self.duplicate(nudged_pdf)
    
        return epsilon
     
    def reorder_variables(self, variables):
        """
        Reorder the variables, for instance if self.numvariables == 3 then call with variables=[2,1,0] to reverse the
        order of the variables. The new joint probability matrix will be determined completely by the old matrix.
        It is also possible to duplicate variables, e.g. variables=[0,1,2,2] to duplicate the last variable (but
        not sure if that is what you want, it will simply copy joint probs, so probably not).
        :param variables: sequence of int
        """
        assert len(variables) >= self.numvariables, 'I cannot reorder if you do\'nt give me the new ordering completely'
        # note: the code is mostly written to support also duplicating a variable

        num_variables_new = len(variables)

        joint_probs_new = np.zeros([self.numvalues]*num_variables_new) - 1

        lists_of_possible_states_per_variable = [range(self.numvalues) for _ in range(num_variables_new)]

        for values_new in itertools.product(*lists_of_possible_states_per_variable):
            values_old_order = [-1]*self.numvariables

            for new_varix in range(len(variables)):
                assert variables[new_varix] < self.numvariables, 'you specified the order of a variable index' \
                                                                 ' >= N (non-existent)'

                # can happen if a variable index is mentioned twice or more in 'variables' but in the current 'values_new'
                # they have different values. This is of course not possible, the two new variables should be equivalent
                # and completely redundant, so then I will set this joint prob. to zero and continue
                if values_old_order[variables[new_varix]] >= 0 \
                        and values_old_order[variables[new_varix]] != values_new[new_varix]:
                    assert len(variables) > self.numvariables, 'at least one variable index should be mentioned twice'

                    joint_probs_new[tuple(values_new)] = 0.0

                    break
                else:
                    # normal case
                    values_old_order[variables[new_varix]] = values_new[new_varix]

            if joint_probs_new[tuple(values_new)] != 0.0:
                assert not -1 in values_old_order, 'missing a variable index in variables=' + str(variables) \
                                                   + ', how should I reorder if you don\'t specify the new order of all' \
                                                   +  ' variables'

                assert joint_probs_new[tuple(values_new)] == -1, 'should still be unset element of joint prob matrix'

                joint_probs_new[tuple(values_new)] = self(values_old_order)
            else:
                pass  # joint prob already set to 0.0 in the above inner loop

        assert not -1 in joint_probs_new, 'not all new joint probs were computed'

        # change myself to the new joint pdf; will also check for being normalized etc.
        self.reset(num_variables_new, self.numvalues, joint_probs_new)