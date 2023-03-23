__author__ = 'rquax'


'''
Owner of this project:

    Rick Quax
    https://staff.fnwi.uva.nl/r.quax
    University of Amsterdam

You are free to use this package only for your own non-profit, academic work. All I ask is to be suitably credited.
'''


import csv
import numpy as np
import itertools
import copy
from scipy.optimize import minimize
import warnings
from compiler.ast import flatten  # note: deprecated in Python 3, in which case: find another flatten
from collections import Sequence
import time
from abc import abstractmethod, ABCMeta  # for requiring certain methods to be overridden by subclasses
from numbers import Integral, Number
import matplotlib.pyplot as plt
import sys
import pathos.multiprocessing as mp
from astroML.plotting import hist  # for Bayesian blocks: automatic determining of variable-size binning of data


# I use this instead of -np.inf as result of log(0), which whould be -inf but then 0 * -inf = NaN, whereas by
# common assumption in information theory: 0 log 0 = 0. So I make it finite here so that the result is indeed 0.
_finite_inf = sys.float_info.max / 1000

def maximum_depth(seq):
    """
    Helper function, e.g. maximum_depth([1,2,[2,4,[[4]]]]) == 4.
    :param seq: sequence, like a list of lists
    :rtype: int
    """
    seq = iter(seq)
    try:
        for level in itertools.count():
            seq = itertools.chain([next(seq)], seq)
            seq = itertools.chain.from_iterable(s for s in seq if isinstance(s, Sequence))
    except StopIteration:
        return level


# helper function,
# from http://stackoverflow.com/questions/2267362/convert-integer-to-a-string-in-a-given-numeric-base-in-python
def int2base(x, b, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    """

    :param x: int
    :type x: int
    :param b: int
    :param b: int
    :param alphabet:
    :rtype : str
    """

    # convert an integer to its string representation in a given base
    if b<2 or b>len(alphabet):
        if b==64: # assume base64 rather than raise error
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        else:
            raise AssertionError("int2base base out of range")

    if isinstance(x,complex): # return a tuple
        return ( int2base(x.real,b,alphabet) , int2base(x.imag,b,alphabet) )

    if x<=0:
        if x==0:
            return alphabet[0]
        else:
            return '-' + int2base(-x,b,alphabet)

    # else x is non-negative real
    rets=''

    while x>0:
        x,idx = divmod(x,b)
        rets = alphabet[idx] + rets

    return str(rets)


def apply_permutation(lst, permutation):
    """
    Return a new list where the element at position ix in <lst> will be at a new position permutation[ix].
    :param lst: list
    :type lst: array_like
    :param permutation:
    :return:
    """
    new_list = [-1]*len(lst)

    assert len(permutation) == len(lst)

    for ix in xrange(len(permutation)):
        new_list[permutation[ix]] = lst[ix]

    if __debug__:
        if not -1 in lst:
            assert not -1 in new_list

    return new_list


# each variable in a JointProbabilityMatrix has a label, and if not provided then this label is used
_default_variable_label = 'variable'


# any derived class from this interface is supposed
# to replace the dictionaries now used for conditional pdfs (namely, dict where keys are
# tuples of values for all variables and the values are JointProbabilityMatrix objects). For now it will be
# made to be superficially equivalent to a dict, then hopefully a derived class can have a different
# inner workings (not combinatorial explosion of the number of keys) while otherwise indistinguishable
# note: use this class name in isinstance(obj, ConditionalProbabilityMatrix)
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
            assert not np.isscalar(cond_pdf.iterkeys().next()), 'should be tuples, even if conditioning on 1 var'

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
        if isinstance(other, basestring):
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
        return self.cond_pdf.iteritems()


    def itervalues(self):
        return self.cond_pdf.itervalues()


    def iterkeys(self):
        return self.cond_pdf.iterkeys()


    def num_given_variables(self):
        assert len(self.cond_pdf) > 0, 'uninitialized dict'

        return len(self.cond_pdf.iterkeys().next())


    def num_output_variables(self):
        assert len(self.cond_pdf) > 0, 'uninitialized dict'

        return len(self.cond_pdf.itervalues().next())


    def num_conditional_probabilities(self):
        return len(self.cond_pdf)


    def update(self, partial_cond_pdf):
        if type(partial_cond_pdf) == dict:
            # check only if already initialized with at least one conditional probability
            if __debug__ and len(self.cond_pdf) > 0:
                assert len(partial_cond_pdf.iterkeys().next()) == self.num_given_variables(), 'partial cond. pdf is ' \
                                                                                              'conditioned on a different' \
                                                                                              ' number of variables'
                assert len(partial_cond_pdf.itervalues().next()) == self.num_output_variables(), \
                    'partial cond. pdf has a different number of output variables'

            self.cond_pdf.update(partial_cond_pdf)
        elif isinstance(partial_cond_pdf, ConditionalProbabilities):
            self.cond_pdf.update(partial_cond_pdf.cond_pdf)
        else:
            raise NotImplementedError('unknown type for partial_cond_pdf')


_type_prob = np.float128

# this class is supposed to override member joint_probabilities of JointProbabilityMatrix, which is currently
# taken to be a nested numpy array. Therefore this class is made to act as much as possible as a nested numpy array.
# However, this class could be inherited and overriden to save e.g. memory storage, e.g. by assuming independence
# among the variables, but on the outside still act the same.
# note: use this class name for isinstance(., .) calls, because it will also be true for derived classes
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


# this class is supposed to override member joint_probabilities of JointProbabilityMatrix, which is currently
# taken to be a nested numpy array. Therefore this class is made to act as much as possible as a nested numpy array.
# However, this class could be inherited and overriden to save e.g. memory storage, e.g. by assuming independence
# among the variables, but on the outside still act the same.
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
                return np.product([self.marginal_probabilities[vix][item[vix]] for vix in xrange(len(item))])
            else:
                assert len(item) < len(self.marginal_probabilities), 'supplied more values than I have variables'

                return IndependentNestedArrayOfProbabilities(np.product([self.marginal_probabilities[vix][item[vix]]
                                                                         for vix in xrange(len(item))])
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
            self.marginal_probabilities = np.array([joint_probabilities.marginalize_distribution([vix])
                                                     .joint_probabilities.matrix2vector()
                                                 for vix in xrange(len(joint_probabilities))], dtype=_type_prob)
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


class CausalImpactResponse(object):
    perturbed_variables = None

    mi_orig = None
    mi_nudged_list = None
    mi_diffs = None
    avg_mi_diff = None
    std_mi_diff = None

    impacts_on_output = None
    avg_impact = None
    std_impact = None

    correlations = None
    avg_corr = None
    std_corr = None

    residuals = None
    avg_residual = None
    std_residual = None

    nudges = None
    upper_bounds_impact = None

    def __init__(self):
        self.perturbed_variables = None

        self.mi_orig = None
        self.mi_nudged_list = None
        self.mi_diffs = None
        self.avg_mi_diff = None
        self.std_mi_diff = None

        self.impacts_on_output = None
        self.avg_impact = None
        self.std_impact = None

        self.correlations = None
        self.avg_corr = None
        self.std_corr = None

        self.residuals = None
        self.avg_residual = None
        self.std_residual = None

        self.nudges = None
        self.upper_bounds_impact = None


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
        elif isinstance(joint_probs, basestring):
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
                numparams = len(self.matrix2params_incremental(True))
                self.params2matrix_incremental(np.random.random(numparams))
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

            np.testing.assert_almost_equal(self.joint_probabilities.sum(), 1.0)
        else:
            pass
            # warnings.warn('numvariables == 0, not sure if it is supported by all code (yet)! Fingers crossed.')


    def copy(self):
        """
        Deep copy.
        :rtype : JointProbabilityMatrix
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
            return [self.generate_sample() for _ in xrange(n)]
        elif method == 'slow':
            def worker_gs(i):
                return self.generate_sample()

            pool = mp.Pool(nprocs)

            ret = pool.map(worker_gs, xrange(n))

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


    def transition_table_deterministic(self, variables_input, variables_output):
        """
        Create a transition table for the given variables, where for each possible sequence of values for the input
        variables I find the values for the output variables which have maximum conditional probability. I marginalize
        first over all other variables not named in either input or output.
        :type variables_input: list of int
        :type variables_output: list of int
        :return: a list of lists, where each sublist is [input_states, max_output_states, max_output_prob], where the
        first two are lists and the latter is a float in [0,1]
        """
        variables = list(variables_input) + list(variables_output)

        pdf_temp = self.marginalize_distribution(variables)

        assert len(pdf_temp) == len(variables)

        trans_table = []

        # for each possible sequence of input states, find the output states which have maximum probability
        for input_states in self.statespace(len(variables_input)):
            max_output_states = []
            max_output_prob = -1.0

            for output_states in self.statespace(len(variables_output)):
                prob = pdf_temp(list(input_states) + list(output_states))

                if prob > max_output_prob:
                    max_output_prob = prob
                    max_output_states = output_states

            # assert max_output_prob >= 1.0 / np.power(self.numvalues, len(variables_output)), \
            #     'how can this be max prob?: ' + str(max_output_prob) + '. I expected at least: ' \
            #     + str(1.0 / np.power(self.numvalues, len(variables_output)))

            trans_table.append([input_states, max_output_states, max_output_prob])

        return trans_table


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


    def random_samples(self, n=1):
        # sample_indices = np.random.multinomial(n, [i for i in self.joint_probabilities.flatiter()])

        # todo: I don't think this generalizes well for different numvalues per variable. Just generate 1000 value lists
        # and pick one according to their relative probabilities? Much more general and above all much more scalable!

        assert np.isscalar(self.numvalues), 'refactor this function, see todo above'

        flat_joint_probs = [i for i in self.joint_probabilities.flatiter()]

        sample_indices = np.random.choice(len(flat_joint_probs), p=flat_joint_probs, size=n)

        values_str_per_sample = [int2base(smpl, self.numvalues).zfill(self.numvariables) for smpl in sample_indices]

        assert len(values_str_per_sample) == n
        assert len(values_str_per_sample[0]) == self.numvariables, 'values_str_per_sample[0] = ' \
                                                                   + str(values_str_per_sample[0])
        assert len(values_str_per_sample[-1]) == self.numvariables, 'values_str_per_sample[-1] = ' \
                                                                   + str(values_str_per_sample[-1])

        values_list_per_sample = [[int(val) for val in valstr] for valstr in values_str_per_sample]

        assert len(values_list_per_sample) == n
        assert len(values_list_per_sample[0]) == self.numvariables
        assert len(values_list_per_sample[-1]) == self.numvariables

        return values_list_per_sample


    def __call__(self, values):
        """
        Joint probability of a list or tuple of values, one value for each variable in order.
        :param values: list of values, each value is an integer in [0, numvalues)
        :type values: list
        :return: joint probability of the given values, in order, for all variables; in [0, 1]
        :rtype: float
        """
        return self.joint_probability(values=values)


    def set_labels(self, labels):
        self.labels = labels


    def get_labels(self):
        return self.labels


    def get_label(self, variable_index):
        return self.labels[variable_index]


    def set_label(self, variable_index, label):
        self.labels[variable_index] = label


    def has_label(self, label):
        return label in self.labels


    def retain_only_labels(self, retained_labels, dict_of_unwanted_label_values):
        unwanted_labels = list(set(self.labels).difference(retained_labels))

        # unwanted_label_values = [dict_of_unwanted_label_values[lbl] for lbl in unwanted_labels]

        unwanted_vars = self.get_variables_with_label_in(unwanted_labels)
        unwanted_var_labels = map(self.get_label, unwanted_vars)
        unwanted_var_values = [dict_of_unwanted_label_values[lbl] for lbl in unwanted_var_labels]

        self.duplicate(self.conditional_probability_distribution(unwanted_vars, unwanted_var_values))



    def clip_all_probabilities(self):
        """
        Make sure all probabilities in the joint probability matrix are in the range [0.0, 1.0], which could be
        violated sometimes due to floating point operation roundoff errors.
        """
        self.joint_probabilities.clip_all_probabilities()


    def estimate_from_data_from_file(self, filename, discrete=True, method='empirical'):
        """
        Same as estimate_from_data but read the data first from the file, then pass data to estimate_from_data.
        :param filename: string, file format will be inferred from the presence of e.g. '.csv'
        :param discrete:
        :param method:
        :raise NotImplementedError: if file format is not recognized
        """
        if '.csv' in filename:
            fin = open(filename, 'rb')
            csvr = csv.reader(fin)

            repeated_measurements = [tuple(row) for row in csvr]

            fin.close()
        else:
            raise NotImplementedError('unknown file format: ' + str(filename))

        self.estimate_from_data(repeated_measurements=repeated_measurements, discrete=discrete, method=method)

    def probs_from_max_numbin(sx):
        # Histogram from list of samples
        d = dict()
        for s in sx:
            d[s] = d.get(s, 0) + 1
        return map(lambda z: float(z) / len(sx), d.values())


    # helper function
    @staticmethod
    def probs_by_bayesian_blocks(array):
        assert len(array) > 0

        if np.isscalar(array[0]):
            bb_bins = hist(array, bins='blocks')
            assert len(bb_bins[0]) > 0
            bb_probs = bb_bins[0] / np.sum(bb_bins[0])

            assert len(bb_probs) > 0
            assert np.isscalar(bb_probs[0])

            return bb_bins[1], bb_probs
        else:
            assert np.rank(array) == 2

            vectors = np.transpose(array)

            return map(JointProbabilityMatrix.probs_by_bayesian_blocks, vectors)


    @staticmethod
    def probs_by_equiprobable_bins(sx, k='auto'):

        if np.isscalar(sx[0]):
            if isinstance(k, basestring):  # like 'auto' or 'freedman', does not really matter
                # k = float(int(np.sqrt(n1)))  # simple heuristic
                # Freedman-Diaconis rule:
                h = 2. * (np.percentile(sx, 75) - np.percentile(sx, 25)) / np.power(len(sx), 1/3.)
                k = int(round((max(sx) - min(sx)) / float(h)))
            else:
                k = float(k)  # make sure it is numeric

            percs = [np.percentile(sx, p) for p in np.linspace(0, 100, k + 1)]
            ##    for i in xrange(n2):
            ##      assert percs[i] == sorted(percs[i]), str('error: np.percentile (#'+str(i)+') did not return a sorted list of values:\n'
            ##                                               +str(percs))
            ##      assert len(percs[i]) == len(set(percs[i])), 'error: entropyd: numbins too large, I get bins of width zero.'
            # remove bins of size zero
            # note: bins of size zero would not contribute to entropy anyway, since 0 log 0 is set
            # to result in 0. For instance, it could be that the data only has 6 unique values occurring,
            # but a numbins=10 is given. Then 4 bins will be empty.
            ##    print('percs: = ' + str(percs))
            percs_fixed = sorted(list(set(percs)))
            ##    probs = np.divide(np.histogramdd(np.hstack(sx), percs)[0], len(sx))
            print('debug: percs_fixed =', percs_fixed)
            probs = np.divide(np.histogramdd(sx, [percs_fixed])[0], len(sx))
            probs1d = list(probs.flat)

            return percs_fixed, probs1d
        else:
            samples_list = np.transpose(sx)

            rets = map(lambda si: JointProbabilityMatrix.probs_by_equiprobable_bins(si, k=k), samples_list)

            return np.transpose(rets)


    @staticmethod
    def convert_continuous_data_to_discrete(samples, numvalues='auto'):

        if np.isscalar(samples[0]):
            if numvalues in ('auto', 'bayes'):
                limits, probs = JointProbabilityMatrix.probs_by_bayesian_blocks(samples)
            else:
                # note: if e.g. k=='freedman' (anything other than 'auto') then a heuristic will be used to compute k
                limits, probs = JointProbabilityMatrix.probs_by_equiprobable_bins(samples, k=numvalues)

            def convert(sample):
                return np.greater_equal(sample, limits[1:-1]).sum()

            return map(convert, samples)
        else:
            samples_list = np.transpose(samples)

            samples_list = map(JointProbabilityMatrix.convert_continuous_data_to_discrete, samples_list)

            return np.transpose(samples_list)


    @staticmethod
    def discretize_data(repeated_measurements, numvalues='auto', return_also_fitness_curve=False, maxnumvalues=20,
                        stopafterdeclines=5, method='equiprobable'):

        pdf = JointProbabilityMatrix(2, 2)  # pre-alloc, values passed are irrelevant

        # assert numvalues == 'auto', 'todo'

        timeseries = np.transpose(repeated_measurements)

        fitness = []  # tuples of (numvals, fitness)

        # print 'debug: len(rep) = %s' % len(repeated_measurements)
        # print 'debug: len(ts) = %s' % len(timeseries)
        # print 'debug: ub =', xrange(2, min(max(int(np.power(len(repeated_measurements), 1.0/2.0)), 5), 100))

        if numvalues == 'auto':
            possible_numvals = xrange(2, min(max(int(np.power(len(repeated_measurements), 1.0/2.0)), 5), maxnumvalues))
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

            print 'debug: processed %s, fitness=%s' % (numvals, fitness_nv)
            if fitness_nv <= 0.0:
                print 'debug: disc. timeseries:', np.transpose(disc_timeseries)[:10]
                print 'debug: repeated_measurements:', np.transpose(repeated_measurements)[:10]
                print 'debug: bounds:', np.transpose(bounds)[:10]

            if True:
                fitness_values = map(lambda x: x[-1], fitness)

                if len(fitness) > 7:
                    if fitness_nv < max(fitness_values) * 0.5:
                        print 'debug: not going to improve, will break the for-loop'
                        break

                if len(fitness) > stopafterdeclines:
                    if list(fitness_values[-stopafterdeclines:]) == sorted(fitness_values[-stopafterdeclines:]):
                        print 'debug: fitness declined %s times in a row, will stop' % stopafterdeclines
                        break

        max_ix = np.argmax(map(lambda x: x[-1], fitness), axis=0)

        if not return_also_fitness_curve:
            return fitness[max_ix][1]
        else:
            return fitness[max_ix][1], map(lambda x: (x[0], x[-1]), fitness)


    def generate_samples_mixed_gaussian(self, n, sigma=0.2, mu=1.):

        if np.isscalar(n):
            samples = self.generate_samples(n)  # each value is integer in 0..numvalues-1 (x below)

            numvals = self.numvalues
        else:
            assert np.ndim(n) == 2, 'expected n to be a list of samples (which are lists of [integer] values)'
            samples = copy.deepcopy(n)

            assert np.min(samples) >= 0, 'sample values are assumed integer values >=0'

            numvals = int(np.max(samples) + 1)

        numvars = np.shape(n)[1]

        # note: assumption is now that the centers and sigmas of each variable are the same
        if np.ndim(mu) < 2:
            mus = [np.arange(numvals) * mu if np.isscalar(mu) else mu for _ in range(numvars)]
        else:
            assert np.ndim(mu) == 2, 'expected mu[varix][valix]'
            mus = mu

        if np.ndim(sigma) < 2:
            sigmas = [[sigma]*numvals if np.isscalar(sigma) else sigma for _ in range(numvars)]
        else:
            assert np.ndim(sigma) == 2, 'expected sigma[varix][valix]'
            sigmas = sigma

        try:
            samples = [[np.random.normal(mus[xix][x], sigmas[xix][x])
                        for xix, x in enumerate(sample)] for sample in samples]
        except IndexError as e:
            print 'debug: np.shape(mus) = %s' % str(np.shape(mus))
            print 'debug: np.shape(sigmas) = %s' % str(np.shape(sigmas))
            print 'debug: np.ndim(mus) = %s' % np.ndim(mu)
            print 'debug: np.ndim(sigmas) = %s' % np.ndim(sigma)
            print 'debug: np.isscalar(mus) = %s' % np.isscalar(mu)
            print 'debug: np.isscalar(sigmas) = %s' % np.isscalar(sigma)
            print 'debug: np.shape(samples) = %s' % str(np.shape(samples))

            raise IndexError(e)

        return samples


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
            all_unique_values = list(set(np.array(repeated_measurements).flat))

            # todo: store the unique values as a member field so that later algorithms know what original values
            # corresponded to the values 0, 1, 2, ... and for estimating MI based on samples rather than 'true' dist.

            if numvalues == 'auto':
                numvals = len(all_unique_values)
            else:
                numvals = int(numvalues)

                for v in xrange(2**31):
                    if len(all_unique_values) < numvals:
                        if not v in all_unique_values:
                            all_unique_values.append(v)  # add bogus values
                    else:
                        break

            dict_val_to_index = {all_unique_values[valix]: valix for valix in xrange(numvals)}

            new_joint_probs = np.zeros([numvals]*numvars)

            # todo: when you generalize self.numvalues to an array then here also set the array instead of int

            for values in repeated_measurements:
                value_indices = tuple((dict_val_to_index[val] for val in values))

                try:
                    new_joint_probs[value_indices] += 1.0 / len(repeated_measurements)
                except IndexError as e:
                    print 'error: value_indices =', value_indices
                    print 'error: type(value_indices) =', type(value_indices)

                    raise IndexError(e)

            self.reset(numvars, numvals, joint_prob_matrix=new_joint_probs)
        elif not discrete and method in ('equiprobable', 'equiprob'):
            disc_timeseries = self.discretize_data(repeated_measurements, numvalues, return_also_fitness_curve=False)

            assert np.shape(repeated_measurements) == np.shape(disc_timeseries)

            self.estimate_from_data(disc_timeseries, numvalues=numvalues, discrete=True, method='empirical')
        else:
            raise NotImplementedError('unknown combination of discrete and method.')


    def loglikelihood(self, repeated_measurements, ignore_zeros=True):
        if not ignore_zeros:
            return np.sum(np.log(map(self, repeated_measurements)))
        else:
            return np.sum(np.log([p for p in map(self, repeated_measurements) if p > 0.]))


    def marginal_probability(self, variables, values):

        if len(set(variables)) == len(self):
            # speedup, save marginalization step, but is functionally equivalent to else clause
            return self.joint_probability(values)
        else:
            return self[variables](values)


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


    def get_variables_with_label_in(self, labels):
        return [vix for vix in xrange(self.numvariables) if self.labels[vix] in labels]


    def marginalize_distribution_retaining_only_labels(self, retained_labels):
        """
        Return a pdf of variables which have one of the labels in the retained_labels set; all other variables will
        be summed out.
        :param retained_labels: list of labels to retain; the rest will be summed over.
        :type retained_labels: sequence
        :rtype: JointProbabilityMatrix
        """

        variable_indices = [vix for vix in xrange(self.numvariables) if self.labels[vix] in retained_labels]

        return self.marginalize_distribution(variable_indices)


    def marginalize_distribution(self, retained_variables):
        """
        Return a pdf of only the given variable indices, summing out all others
        :param retained_variables: variables to retain, all others will be summed out and will not be a variable anymore
        :type: array_like
        :rtype : JointProbabilityMatrix
        """
        lists_of_possible_states_per_variable = [range(self.numvalues) for variable in xrange(self.numvariables)]

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

            np.testing.assert_almost_equal(np.sum(marginalized_joint_probs), 1.0)

            marginal_joint_pdf = JointProbabilityMatrix(len(retained_variables), self.numvalues,
                                                        joint_probs=marginalized_joint_probs)

            return marginal_joint_pdf


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
                        for val in xrange(self.numvalues)]
            else:
                # same as other case but np.array converted, since the joint pdf matrix is always expected to be that
                return np.array([self.appended_joint_prob_matrix(num_added_variables,
                                                               values_so_far=list(values_so_far) + [val],
                                                               added_joint_probabilities=added_joint_probabilities)
                                 for val in xrange(self.numvalues)])
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

        lists_of_possible_given_values = [range(self.numvalues) for _ in xrange(self.numvariables)]

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
        np.testing.assert_almost_equal(np.sum(extended_joint_probs), 1.0)

        self.reset(len(state_transitions[0]), self.numvalues, extended_joint_probs)


    def reverse_reordering_variables(self, variables):

        varlist = list(variables)
        numvars = self.numvariables

        # note: if a ValueError occurs then you did not specify every variable index in <variables>, like
        # [1,2] instead of [1,2,0].
        reverse_ordering = [varlist.index(ix) for ix in xrange(numvars)]

        self.reorder_variables(reverse_ordering)



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

        lists_of_possible_states_per_variable = [range(self.numvalues) for _ in xrange(num_variables_new)]

        for values_new in itertools.product(*lists_of_possible_states_per_variable):
            values_old_order = [-1]*self.numvariables

            for new_varix in xrange(len(variables)):
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


    # todo: implement __sub__, but for this it seems necessary to give each variable a unique ID at creation (__init__)
    # and keep track of them as you do operations such as marginalizing. Otherwise, subtraction is ambiguous,
    # i.e., subtracting a 2-variable pdf from a 5-variable pdf should result in a 3-variable conditional pdf, but
    # it is not clear which 3 variables should be chosen. Can choose this to always be the first 3 and let the
    # user be responsible to reorder them before subtraction, but seems prone to error if user does not know this or
    # forgets? Well can test equality by marginalizing the 5-variable pdf...
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


    # todo: implement __setitem__ for either pdf or cond. pdfs


    def __len__(self):
        return self.numvariables


    def __getitem__(self, item):
        if item == 'all':
            return self
        elif not hasattr(item, '__iter__'):
            return self.marginalize_distribution([item])
        else:
            return self.marginalize_distribution(item)


    def __iadd__(self, other):
        """

        :param other: can be JointProbabilityMatrix or a conditional distribution (dict of JointProbabilityMatrix)
        :type other: JointProbabilityMatrix or ConditionalProbabilities or dict
        """

        self.append_variables_using_conditional_distributions(other)


    def matrix2vector(self):
        return self.joint_probabilities.flatten()


    def vector2matrix(self, list_probs):
        np.testing.assert_almost_equal(np.sum(list_probs), 1.0)

        assert np.ndim(list_probs) == 1

        self.joint_probabilities.reset(np.reshape(list_probs, [self.numvalues]*self.numvariables))

        self.clip_all_probabilities()


    def params2matrix(self, parameters):
        assert len(parameters) == np.power(self.numvalues, self.numvariables) - 1

        vector_probs = [-1.0]*(np.power(self.numvalues, self.numvariables))

        remaining_prob_mass = 1.0

        for pix in xrange(len(parameters)):
            # note: small rounding errors will be fixed below by clipping
            assert -0.000001 <= parameters[pix] <= 1.000001, 'parameters should be in [0, 1]: ' + str(parameters[pix])

            # clip the parameter to the allowed range. If a rounding error is fixed by this in the parameters then
            # possibly a rounding error will appear in the probabilities?... Not sure though
            parameters[pix] = min(max(parameters[pix], 0.0), 1.0)

            vector_probs[pix] = remaining_prob_mass * parameters[pix]

            remaining_prob_mass = remaining_prob_mass * (1.0 - parameters[pix])

        assert vector_probs[-1] < 0.0, 'should still be unset by the above loop'

        # last parameter is irrelevant, must always be 1.0 is also a way to look at it
        vector_probs[-1] = remaining_prob_mass

        np.testing.assert_almost_equal(np.sum(vector_probs), 1.0)

        self.vector2matrix(vector_probs)


    def from_params(self, parameters):  # alternative constructor
        self.params2matrix(parameters)

        return self


    def matrix2params(self):
        vector_probs = self.matrix2vector()

        remaining_prob_mass = 1.0

        parameters = [-1.0]*(len(vector_probs) - 1)

        np.testing.assert_almost_equal(np.sum(vector_probs), 1.0)

        for pix in xrange(len(parameters)):
            if remaining_prob_mass > 0:
                assert remaining_prob_mass <= 1.0, 'remaining prob mass: ' + str(remaining_prob_mass)
                assert vector_probs[pix] <= remaining_prob_mass + 0.00001, \
                    'vector_probs[pix]=' + str(vector_probs[pix]) + ', remaining_prob_mass=' + str(remaining_prob_mass)

                parameters[pix] = vector_probs[pix] / remaining_prob_mass

                assert -0.1 <= parameters[pix] <= 1.1, \
                    'parameters should be in [0, 1]: ' + str(parameters[pix]) \
                    + ', sum probs = ' + str(np.sum(self.joint_probabilities.joint_probabilities))

                # sometimes this happens I think due to rounding errors, but when I sum the probabilities they
                # still seem to sum to exactly 1.0 so probably is due to some parameters being 0 or 1, so clip here
                # parameters[pix] = max(min(parameters[pix], 1.0), 0.0)  # is already below
            elif remaining_prob_mass == 0:
                parameters[pix] = 0
            else:
                if not remaining_prob_mass > -0.000001:
                    print 'debug: remaining_prob_mass =', remaining_prob_mass
                    print 'debug: pix =', pix, 'out of', len(parameters)

                    # todo: if just due to floating operation error, so small, then clip to zero and go on?
                    raise ValueError('remaining_prob_mass = ' + str(remaining_prob_mass)
                                     + ' < 0, which should not happen?')
                else:
                    # seems that it was intended to reach zero but due to floating operation roundoff it got just
                    # slightly under. Clip to 0 will do the trick.
                    remaining_prob_mass = 0.0  # clip to zero, so it will stay that way

                parameters[pix] = 0  # does not matter

                assert -0.1 <= parameters[pix] <= 1.1, \
                    'parameters should be in [0, 1]: ' + str(parameters[pix]) \
                    + ', sum probs = ' + str(np.sum(self.joint_probabilities.joint_probabilities))

            # sometimes this happens I think due to rounding errors, but when I sum the probabilities they
            # still seem to sum to exactly 1.0 so probably is due to some parameters being 0 or 1, so clip here
            parameters[pix] = max(min(parameters[pix], 1.0), 0.0)

            # parameters[pix] = min(max(parameters[pix], 0.0), 1.0)

            remaining_prob_mass -= remaining_prob_mass * parameters[pix]

        return parameters


    def __add__(self, other):
        """
        Append the variables defined by the (conditional) distributions in other.
        :type other: dict of JointProbabilityMatrix | JointProbabilityMatrix
        :rtype: JointProbabilityMatrix
        """

        pdf = self.copy()
        pdf.append_variables_using_conditional_distributions(other)

        return pdf

    def matrix2params_incremental(self, return_flattened=True, verbose=False):
        if self.numvariables > 1:
            # get the marginal pdf for the first variable
            pdf1 = self.marginalize_distribution([0])

            # first sequence of parameters, rest is added below here
            parameters = pdf1.matrix2params()

            pdf_conds = self.conditional_probability_distributions([0])

            # assert len(pdf_conds) == self.numvalues, 'should be one pdf for each value of first variable'

            for val in xrange(self.numvalues):
                pdf_cond = pdf_conds[tuple([val])]

                added_params = pdf_cond.matrix2params_incremental(return_flattened=False, verbose=verbose)

                if verbose:
                    print 'debug: matrix2params_incremental: recursed: for val=' + str(val) + ' I got added_params=' \
                          + str(added_params) + '.'
                    print 'debug: matrix2params_incremental: old parameters =', parameters

                # instead of returning a flat list of parameters I make it nested, so that the structure (e.g. number of
                # variables and number of values) can be inferred, and also hopefully it can be inferred to which
                # variable which parameters belong.
                # CHANGE123
                parameters.append(added_params)

                if verbose:
                    print 'debug: matrix2params_incremental: new parameters =', parameters

            if return_flattened:
                # flatten the tree structure to a list of scalars, which is sorted on the variable id
                parameters = self.scalars_up_to_level(parameters)

            return parameters
        elif self.numvariables == 1:
            return self.matrix2params()
        else:
            raise ValueError('no parameters for 0 variables')

    _debug_params2matrix = False  # internal variable, used to debug a debug statement, can be removed in a while


    # todo: should this be part of e.g. FullNestedArrayOfProbabilities instead of this class?
    def params2matrix_incremental(self, parameters):
        """
        Takes in a row of floats in range [0.0, 1.0] and changes <self> to a new PDF which is characterized by the
        parameters. Benefit: np.random.random(M**N - 1) results in an unbiased sample of PDF, wnere M is numvalues
        and N is numvariables.
        :param parameters: list of floats, length equal to what matrix2params_incrmental() returns (M**N - 1)
        :type parameters: list of float
        """
        if __debug__:
            # store the original provided list of scalars
            original_parameters = list(parameters)

        # I suspect that both a tree-like input and a list of scalars should work... (add to unit test?)
        if np.all(map(np.isscalar, parameters)):
            assert min(parameters) > -0.0001, 'parameter(s) significantly out of allowed bounds [0,1]: ' \
                                              + str(parameters)
            assert min(parameters) < 1.0001, 'parameter(s) significantly out of allowed bounds [0,1]: ' \
                                              + str(parameters)

            # clip each parameter value to the allowed range. above I check already whether the error is not too large
            parameters = [min(max(pi, 0.0), 1.0) for pi in parameters]

            parameters = self.imbalanced_tree_from_scalars(parameters, self.numvalues)

            # verify that the procedure to make the tree out of the list of scalars is reversible and correct
            # (looking for bug)
            if __debug__ and self._debug_params2matrix:
                original_parameters2 = self.scalars_up_to_level(parameters)

                np.testing.assert_array_almost_equal(original_parameters, original_parameters2)

        if self.numvariables > 1:
            # first (numvalues - 1) values in the parameters tree structure should be scalars, as they will be used
            # to make the first variable's marginal distribution
            assert np.all(map(np.isscalar, parameters[:(self.numvalues - 1)]))

            ### start already by setting the pdf of the first variable...

            pdf_1 = JointProbabilityMatrix(1, self.numvalues)
            pdf_1.params2matrix(parameters[:(len(pdf_1.joint_probabilities.flatiter()) - 1)])

            assert (len(pdf_1.joint_probabilities.flatiter()) - 1) == (self.numvalues - 1), 'assumption directly above'

            assert len(pdf_1.joint_probabilities.flatiter()) == self.numvalues

            assert len(flatten(parameters)) == len(self.joint_probabilities.flatiter()) - 1, \
                'more or fewer parameters than needed: ' \
                  'need ' + str(len(self.joint_probabilities.flatiter()) - 1) + ', got ' + str(len(flatten(parameters))) \
                  + '; #vars, #vals = ' + str(self.numvariables) + ', ' + str(self.numvalues)

            if __debug__ and self._debug_params2matrix:
                # remove this (expensive) check after it seems to work a few times?
                # note: for the conditions of no 1.0 or 0.0 prior probs, see the note in params2matrix_incremental
                if not 0.0 in pdf_1.matrix2params() and not 1.0 in pdf_1.matrix2params():
                    np.testing.assert_array_almost_equal(pdf_1.matrix2params(),
                                                         self.scalars_up_to_level(parameters[:(self.numvalues - 1)]))
                    np.testing.assert_array_almost_equal(pdf_1.matrix2params_incremental(),
                                                         self.scalars_up_to_level(parameters[:(self.numvalues - 1)]))

            # remove the used parameters from the list
            parameters = parameters[(len(pdf_1.joint_probabilities.flatiter()) - 1):]
            assert len(parameters) == self.numvalues  # one subtree per conditional pdf

            pdf_conds = dict()

            ### now add other variables...

            for val in xrange(self.numvalues):
                # set this conditional pdf recursively as defined by the next sequence of parameters
                pdf_cond = JointProbabilityMatrix(self.numvariables - 1, self.numvalues)

                # note: parameters[0] is a sublist
                assert not np.isscalar(parameters[0])

                assert not np.isscalar(parameters[0])

                # todo: changing the parameters list is not necessary, maybe faster if not?

                # pdf_cond.params2matrix_incremental(parameters[:(len(pdf_cond.joint_probabilities.flatiter()) - 1)])
                pdf_cond.params2matrix_incremental(parameters[0])

                # conditional pdf should have the same set of parameters as the ones I used to create it
                # (todo: remove this expensive check if it seems to work for  while)
                if self._debug_params2matrix:  # seemed to work for a long time...
                    try:
                        if np.random.randint(20) == 0:
                            np.testing.assert_array_almost_equal(pdf_cond.matrix2params_incremental(),
                                                                 self.scalars_up_to_level(parameters[0]))
                    except AssertionError as e:
                        # print 'debug: parameters[0] =', parameters[0]
                        # print 'debug: len(pdf_cond) =', len(pdf_cond)
                        # print 'debug: pdf_cond.joint_probabilities =', pdf_cond.joint_probabilities

                        pdf_1_duplicate1 = pdf_cond.copy()
                        pdf_1_duplicate2 = pdf_cond.copy()

                        pdf_1_duplicate1._debug_params2matrix = False  # prevent endless recursion
                        pdf_1_duplicate2._debug_params2matrix = False  # prevent endless recursion

                        pdf_1_duplicate1.params2matrix_incremental(self.scalars_up_to_level(parameters[0]))
                        pdf_1_duplicate2.params2matrix_incremental(pdf_cond.matrix2params_incremental())

                        pdf_1_duplicate1._debug_params2matrix = True
                        pdf_1_duplicate2._debug_params2matrix = True

                        assert pdf_1_duplicate1 == pdf_cond
                        assert pdf_1_duplicate2 == pdf_cond

                        del pdf_1_duplicate1, pdf_1_duplicate2

                        # note: the cause seems to be as follows. If you provide the parameters e.g.
                        # [0.0, [0.37028884415935004], [0.98942830522914993]] then the middle parameter is superfluous,
                        # because it defines a conditional probability p(b|a) for which its prior p(a)=0. So no matter
                        # what parameter is here, the joint prob p(a,b) will be zero anyway. In other words, many
                        # parameter lists map to the same p(a,b). This makes the inverse problem ambiguous:
                        # going from a p(a,b) to a parameter list. So after building a pdf from the above example of
                        # parameter values I may very well get a different parameter list from that pdf, even though
                        # the pdf built is the one intended. I don't see a way around this because even if this class
                        # makes it uniform, e.g. always making parameter values 0.0 in case their prior is zero,
                        # but then still a user or optimization procedure can provide any list of parameters, so
                        # then also the uniformized parameter list will differ from the user-supplied.

                        # raise AssertionError(e)

                        # later add check. If this check fails then for sure there is something wrong. See also the
                        # original check below.
                        assert 0.0 in self.scalars_up_to_level(parameters) or \
                               1.0 in self.scalars_up_to_level(parameters), 'see story above. ' \
                                                                               'self.scalars_up_to_level(parameters) = ' \
                                                                               + str(self.scalars_up_to_level(parameters))

                        # original check. This check failed once, but the idea was to see if there are 0s or 1s in the
                        # prior probability distribution, which precedes the conditional probability distribution for which
                        # apparently the identifying parameter values have changed. But maybe I am wrong in that
                        # parameters[0] is not the prior only, and some prior prob. information is in all of parameters,
                        # I am not sure anymore so I added the above check to see whether that one is hit instead
                        # of this one (so above check is of course more stringent than this one....)
                        # assert 0.0 in self.scalars_up_to_level(parameters[0]) or \
                        #        1.0 in self.scalars_up_to_level(parameters[0]), 'see story above. ' \
                        #                                                        'self.scalars_up_to_level(parameters[0]) = ' \
                        #                                                        + str(self.scalars_up_to_level(parameters[0]))

                np.testing.assert_almost_equal(pdf_cond.joint_probabilities.sum(), 1.0)

                parameters = parameters[1:]

                # add the conditional pdf
                pdf_conds[(val,)] = pdf_cond.copy()

            assert len(parameters) == 0, 'all parameters should be used to construct joint pdf'

            pdf_1.append_variables_using_conditional_distributions(pdf_conds)

            if __debug__ and self._debug_params2matrix:
                # remove this (expensive) check after it seems to work a few times?
                try:
                    np.testing.assert_array_almost_equal(pdf_1.matrix2params_incremental(),
                                                         self.scalars_up_to_level(original_parameters))
                except AssertionError as e:
                    ### I have the hunch that the above assertion is hit but that it is only if a parameter is 1 or 0,
                    ### so that the parameter may be different but that it does not matter. still don't understand
                    ### why it happens though...

                    pdf_1_duplicate = pdf_1.copy()

                    pdf_1_duplicate._debug_params2matrix = False  # prevent endless recursion

                    pdf_1_duplicate.params2matrix_incremental(self.scalars_up_to_level(original_parameters))

                    pdf_1_duplicate._debug_params2matrix = True

                    if not pdf_1_duplicate == pdf_1:
                        print 'error: the pdfs from the two different parameter lists are also not equivalent'

                        del pdf_1_duplicate

                        raise AssertionError(e)
                    else:
                        # warnings.warn('I found that two PDF objects can have the same joint prob. matrix but a'
                        #               ' different list of identifying parameters. This seems to be due to a variable'
                        #               ' having 0.0 probability on a certain value, making the associated conditional'
                        #               ' PDF of other variables 0 and therefore those associated parameters irrelevant.'
                        #               ' Find a way to make these parameters still uniform? Seems to happen in'
                        #               ' "pdf_1.append_variables_using_conditional_distributions(pdf_conds)"...')

                        # note: (duplicated) the cause seems to be as follows. If you provide the parameters e.g.
                        # [0.0, [0.37028884415935004], [0.98942830522914993]] then the middle parameter is superfluous,
                        # because it defines a conditional probability p(b|a) for which its prior p(a)=0. So no matter
                        # what parameter is here, the joint prob p(a,b) will be zero anyway. In other words, many
                        # parameter lists map to the same p(a,b). This makes the inverse problem ambiguous:
                        # going from a p(a,b) to a parameter list. So after building a pdf from the above example of
                        # parameter values I may very well get a different parameter list from that pdf, even though
                        # the pdf built is the one intended. I don't see a way around this because even if this class
                        # makes it uniform, e.g. always making parameter values 0.0 in case their prior is zero,
                        # but then still a user or optimization procedure can provide any list of parameters, so
                        # then also the uniformized parameter list will differ from the user-supplied.

                        del pdf_1_duplicate

            assert pdf_1.numvariables == self.numvariables
            assert pdf_1.numvalues == self.numvalues

            self.duplicate(pdf_1)  # make this object (self) be the same as pdf_1
        elif self.numvariables == 1:
            self.params2matrix(parameters)
        else:
            assert len(parameters) == 0, 'at the least 0 parameters should be given for 0 variables...'

            raise ValueError('no parameters for 0 variables')


    def duplicate(self, other_joint_pdf):
        """

        :type other_joint_pdf: JointProbabilityMatrix
        """
        self.reset(other_joint_pdf.numvariables, other_joint_pdf.numvalues, other_joint_pdf.joint_probabilities)


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
        #                                                         'self.numvariables = ' + str(self.numvariables) + ', ' \
        #                                                         'joint matrix = ' + str(joint_prob_matrix)
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


    def statespace(self, numvars='all'):
        if numvars == 'all':
            lists_of_possible_joint_values = [range(self.numvalues) for _ in xrange(self.numvariables)]
        elif type(numvars) in (int,):
            lists_of_possible_joint_values = [range(self.numvalues) for _ in xrange(numvars)]
        else:
            raise NotImplementedError('numvars=' + str(numvars))

        return itertools.product(*lists_of_possible_joint_values)


    def append_redundant_variables(self, num_redundant_variables):
        self.append_variables_using_state_transitions_table(lambda my_values_tuple, numvals:
                                                            [sum(my_values_tuple) % numvals]*num_redundant_variables)


    def append_copy_of(self, variables):
        self.append_variables_using_state_transitions_table(lambda vals, nv: vals)
        # note to self: p(A)*p(B|A) == p(B)*p(A|B) leads to: p(A)/p(B) == p(A|B)/p(B|A)
        #   -->  p(A)/p(B) == [p(A)*p(B|A)/p(B)]/p(B|A)


    def append_nudged(self, variables, epsilon=0.01, num_repeats=5, verbose=False):
        pdf_P = self[variables]
        pdf_P_copy = pdf_P.copy()
        nudge = pdf_P_copy.nudge(range(len(variables)), [], epsilon=epsilon)
        desired_marginal_probabilities = pdf_P_copy.joint_probabilities.joint_probabilities

        # a global variable used by the cost_func() to use for quantifying the cost, without having to re-create a new
        # pdf object every time
        pdf_new = pdf_P.copy()
        pdf_new.append_variables(len(variables))

        max_mi = pdf_P.entropy()

        def cost_func(free_params, parameter_values_before):
            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))
            pdf_P_nudged = pdf_new[range(int(len(pdf_new) / 2), len(pdf_new))]  # marginalize the nudged version
            prob_diffs = pdf_P_nudged.joint_probabilities.joint_probabilities - desired_marginal_probabilities

            cost_marginal = np.sum(np.power(prob_diffs, 2))
            # percentage reduction in MI, to try to bring to the same order of magnitude as cost_marginal
            cost_mi = (max_mi - pdf_new.mutual_information(range(int(len(pdf_new) / 2)),
                                                           range(int(len(pdf_new) / 2), len(pdf_new)))) / max_mi
            cost_mi = epsilon * np.power(cost_mi, 2)
            return cost_marginal + cost_mi

        pdf_P.append_optimized_variables(len(variables), cost_func, num_repeats=num_repeats, verbose=verbose)

        # get the actually achieved marginal probabilities of the append variables, which are hopefully close to
        # desired_marginal_probabilities
        obtained_marginal_probabilities = pdf_P[range(len(variables), 2*len(variables))].joint_probabilities.joint_probabilities
        # the array obtained_nudge is hopefully very close to ``nudge''
        obtained_nudge = obtained_marginal_probabilities - pdf_P[range(len(variables))].joint_probabilities.joint_probabilities

        cond_P_nudged = pdf_P.conditional_probability_distributions(range(len(variables)),
                                                                    range(len(variables), 2*len(variables)))

        self.append_variables_using_conditional_distributions(cond_P_nudged, variables)

        return nudge, obtained_nudge


    def append_nudged_direct(self, variables, epsilon=0.01):
        self.append_copy_of(variables)
        nudge = self.nudge(variables, np.setdiff1d(range(len(self)), variables), epsilon=epsilon)

        print 'debug: MI:', self.mutual_information(variables, range(len(self) - len(variables), len(self)))

        ordering = range(len(self))
        for vix in xrange(len(variables)):
            ordering[variables[vix]] = ordering[len(self) - len(variables) + vix]
            ordering[len(self) - len(variables) + vix] = variables[vix]

        self.reorder_variables(ordering)

        return nudge


    def append_independent_variables(self, joint_pdf):
        """

        :type joint_pdf: JointProbabilityMatrix
        """
        assert not type(joint_pdf) in (int, float, str), 'should pass a JointProbabilityMatrix object'

        self.append_variables_using_conditional_distributions({(): joint_pdf})


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
                    for debug_pdf in cond_pdf.itervalues():
                        assert isinstance(debug_pdf, JointProbabilityMatrix), 'debug_pdf = ' + str(debug_pdf)

                statespace_per_independent_variable = [range(self.numvalues)
                                                       for _ in xrange(num_independent_vars)]

                if given_variables is None:
                    for indep_vals in itertools.product(*statespace_per_independent_variable):
                        pdf_conds_complete.update({(tuple(indep_vals) + tuple(key)): value
                                                   for key, value in cond_pdf.iteritems()})
                else:
                    assert len(given_variables) == len(cond_pdf.iterkeys().next()), \
                        'if conditioned on ' + str(len(given_variables)) + 'then I also expect a conditional pdf ' \
                        + 'which conditions on ' + str(len(given_variables)) + ' variables.'

                    not_given_variables = np.setdiff1d(range(self.numvariables), given_variables)

                    assert len(not_given_variables) + len(given_variables) == self.numvariables

                    for indep_vals in itertools.product(*statespace_per_independent_variable):
                        pdf_conds_complete.update({tuple(apply_permutation(indep_vals + tuple(key),
                                                                           list(not_given_variables)
                                                                           + list(given_variables))): value
                                                   for key, value in cond_pdf.iteritems()})

                assert len(pdf_conds_complete.iterkeys().next()) == len(self)
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
        if __debug__ and len(self) == 1:
            _debug_parameters_before_append = self.matrix2params()

            # note: in the loop below the sublists of parameters will be added

        statespace_per_variable = [range(self.numvalues)
                                          for _ in xrange(self.numvariables + num_added_variables)]

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

            if __debug__ and len(self) == 1:
                _debug_parameters_before_append.append(cond_pdf[tuple(existing_variables_values)].matrix2params_incremental)

        if __debug__ and np.random.randint(10) == 0:
            np.testing.assert_almost_equal(np.sum(extended_joint_probs), 1.0)

        self.reset(self.numvariables + num_added_variables, self.numvalues, extended_joint_probs)

        # if __debug__ and len(self) == 1:
        #     # of course this test depends on the implementation of matrix2params_incremental, currently it should
        #     # work
        #     np.testing.assert_array_almost_equal(self.scalars_up_to_level(_debug_parameters_before_append),
        #                                          self.matrix2params_incremental(return_flattened=True))


    def conditional_probability_distribution(self, given_variables, given_values):
        """

        :param given_variables: list of integers
        :param given_values: list of integers
        :rtype: JointProbabilityMatrix
        """
        assert len(given_values) == len(given_variables)
        assert len(given_variables) < self.numvariables, 'no variables left after conditioning'

        lists_of_possible_states_per_variable = [range(self.numvalues) for variable in xrange(self.numvariables)]

        # overwrite the 'state spaces' for the specified variables, to the specified state spaces
        for gix in xrange(len(given_variables)):
            assert np.isscalar(given_values[gix]), 'assuming specific value, not list of possibilities'

            lists_of_possible_states_per_variable[given_variables[gix]] = \
                [given_values[gix]] if np.isscalar(given_values[gix]) else given_values[gix]

        conditioned_variables = [varix for varix in xrange(self.numvariables) if not varix in given_variables]

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
                print 'debug: conditional_probs =', conditional_probs
                print 'debug: min(conditional_probs) =', min(conditional_probs)

                raise ValueError(e)

            np.testing.assert_almost_equal(np.max(conditional_probs), 0.0)

            conditional_probs *= 0
            conditional_probs += 1.0  # create some fake mass, making it a uniform distribution

            conditional_probs /= np.sum(conditional_probs)

        conditional_joint_pdf = JointProbabilityMatrix(len(conditioned_variables), self.numvalues,
                                                    joint_probs=conditional_probs)

        return conditional_joint_pdf


    def conditional_probability_distributions(self, given_variables, object_variables='auto', nprocs=1):
        """

        :param given_variables:
        :return: dict of JointProbabilityMatrix, keys are all possible values for given_variables
        :rtype: dict of JointProbabilityMatrix
        """
        if len(given_variables) == self.numvariables:  # 'no variables left after conditioning'
            warnings.warn('conditional_probability_distributions: no variables left after conditioning')

            lists_of_possible_given_values = [range(self.numvalues) for variable in xrange(len(given_variables))]

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

            lists_of_possible_given_values = [range(self.numvalues) for variable in xrange(len(given_variables))]

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


    # todo: try to make this entropy() function VERY efficient, maybe compiled (weave?), or C++ binding or something,
    # it is a central
    # function in all information-theoretic quantities and especially it is a crucial bottleneck for optimization
    # procedures such as in append_orthogonal* which call information-theoretic quantities a LOT.
    def entropy(self, variables=None, base=2):

        if variables is None:
            if np.random.random() < 0.1:
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


    def conditional_entropy(self, variables, given_variables=None):
        assert hasattr(variables, '__iter__'), 'variables1 = ' + str(variables)
        assert hasattr(given_variables, '__iter__') or given_variables is None, 'variables2 = ' + str(given_variables)

        assert max(variables) < self.numvariables, 'variables are 0-based indices, so <= N - 1: variables=' \
                                                   + str(variables) + ' (N=' + str(self.numvariables) + ')'

        if given_variables is None:
            # automatically set the given_variables to be the complement of 'variables', so all remaining variables

            given_variables = [varix for varix in xrange(self.numvariables) if not varix in variables]

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

            given_variables = [varix for varix in xrange(self.numvariables)
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

            given_variables = [varix for varix in xrange(self.numvariables)
                               if not varix in list(variables1) + list(variables2)]

        condmis = self.conditional_mutual_informations(variables1, variables2, given_variables)
        p3 = self.marginalize_distribution(given_variables)

        return sum([p3(z) * condmis[z] for z in p3.statespace()])


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


    def unique_individual_information(self, variables_Y, variables_X, tol_nonunq=0.05, verbose=True,
                                      num_repeats_per_append=3, assume_independence=False):

        """

        :param variables_Y:
        :param variables_X:
        :param tol_nonunq:
        :param verbose:
        :param num_repeats_per_append:
        :param assume_independence: if all variables_X are independent then this function greatly simplifies to the
        sum of mutual information terms.
        :return:
        """

        if assume_independence:
            return sum([self.mutual_information(variables_Y, [x]) for x in variables_X])
        else:
            pdf_c = self.copy()
            pdf_X = self.marginalize_distribution(variables_X)

            xixs = np.random.permutation(range(len(variables_X)))

            for xid in xrange(len(xixs)):
                pdf_X.append_unique_individual_variable(xixs[xid], verbose=verbose, tol_nonunique=tol_nonunq,
                                                        num_repeats=num_repeats_per_append,
                                                        ignore_variables=(None if xid <= 0 else xixs[:xid]))
                # note: agnostic_about is only set in case there are at least two other unique variables since
                # in that case there could be synergistic information about other unique information. For only one
                # other unique variable added this is not the case and the optimization itself already discounts for
                # unique information about others
                # CHECK: actually it discounts the total information with all other variables so

            # append the URVs to the original pdf_c so that we can compute MI(Y:URVs)
            cond_urvs = pdf_X.conditional_probability_distributions(range(len(variables_X)))
            pdf_c.append_variables_using_conditional_distributions(cond_urvs, variables_X)

            # sum up individual MI's with URVs. If done all URVs simultaneously then synergistic information arises,
            # like Y=XOR(X1,X2) would have 1.0 unique information then (X1,X2 independent).
            mi = np.sum([pdf_c.mutual_information([urvix], variables_Y) for urvix in range(len(self), len(pdf_c))])

            return mi


    # todo: return not just a number but an object with more result information, which maybe if evaluated as
    # float then it will return the current return value
    def synergistic_information(self, variables_Y, variables_X, tol_nonsyn_mi_frac=0.05, verbose=True,
                                minimize_method=None, num_repeats_per_srv_append=3, initial_guess_summed_modulo=False):

        pdf_with_srvs = self.copy()

        # TODO: improve the retrying of finding and adding srvs, in different orders?

        syn_entropy = self.synergistic_entropy_upper_bound(variables_X)
        max_ent_per_var = np.log2(self.numvalues)
        max_num_srv_add_trials = int(round(syn_entropy / max_ent_per_var * 2 + 0.5))  # heuristic
        # ent_X = self.entropy(variables_X)

        # note: currently I constrain to add SRVs which consist each of only 1 variable each. I guess in most cases
        # this will work fine, but I do not know for sure that this will always be sufficient (i.e., does there
        # ever an MSRV exist with bigger entropy, and if so, is this MSRV not decomposable into parts such that
        # this approach here still works?)

        total_syn_mi = 0

        for i in xrange(max_num_srv_add_trials):
            try:
                agnostic_about = range(len(self), len(pdf_with_srvs))  # new SRV must not correlate with previous SRVs
                pdf_with_srvs.append_synergistic_variables(1, initial_guess_summed_modulo=initial_guess_summed_modulo,
                                                           subject_variables=variables_X,
                                                           num_repeats=num_repeats_per_srv_append,
                                                           agnostic_about=agnostic_about,
                                                           minimize_method=minimize_method)
            except UserWarning as e:
                assert 'minimize() failed' in str(e), 'only known reason for this error'

                warnings.warn(str(e) + '. Will now skip this sample in synergistic_information.')

                continue

            total_mi = pdf_with_srvs.mutual_information([-1], variables_X)
            indiv_mi_list =  [pdf_with_srvs.mutual_information([-1], [xid]) for xid in variables_X]
            new_syn_info = total_mi - sum(indiv_mi_list)  # lower bound, actual amount might be higher (but < total_mi)

            if new_syn_info < syn_entropy * 0.01:  # very small added amount if syn. info., so stop after this one
                if (total_mi - new_syn_info) / total_mi > tol_nonsyn_mi_frac:  # too much non-syn. information?
                    pdf_with_srvs.marginalize_distribution(range(len(pdf_with_srvs) - 1))  # remove last srv

                    if verbose:
                        print 'debug: synergistic_information: a last SRV was found but with too much non-syn. info.'
                else:
                    if verbose:
                        print 'debug: synergistic_information: a last (small) ' \
                              'SRV was found, a good one, and now I will stop.'

                    total_syn_mi += new_syn_info

                break  # assume that no more better SRVs can be found from now on, so stop (save time)
            else:
                if i == max_num_srv_add_trials - 1:
                    warnings.warn('synergistic_information: never converged to adding SRV with ~zero synergy')

                if (total_mi - new_syn_info) / total_mi > tol_nonsyn_mi_frac:  # too much non-synergistic information?
                    pdf_with_srvs.marginalize_distribution(range(len(pdf_with_srvs) - 1))  # remove last srv

                    if verbose:
                        print('debug: synergistic_information: an SRV with new_syn_info/total_mi = '
                              + str(new_syn_info) + ' / ' + str(total_mi) + ' = '
                              + str((new_syn_info) / total_mi) + ' was found, which will be removed again because'
                                                                     ' it does not meet the tolerance of '
                              + str(tol_nonsyn_mi_frac))
                else:
                    if verbose:
                        print 'debug: synergistic_information: an SRV with new_syn_info/total_mi = ' \
                              + str(new_syn_info) + ' / ' + str(total_mi) + ' = ' \
                                  + str((new_syn_info) / total_mi) + ' was found, which satisfies the tolerance of ' \
                                  + str(tol_nonsyn_mi_frac)

                        if len(agnostic_about) > 0:
                            agn_mi = pdf_with_srvs.mutual_information([-1], agnostic_about)

                            print 'debug: agnostic=%s, agn. mi = %s (should be close to 0)' % (agnostic_about, agn_mi)

                    total_syn_mi += new_syn_info

            if total_syn_mi >= syn_entropy * 0.99:
                if verbose:
                    print 'debug: found %s\% of the upper bound of synergistic entropy which is high so I stop.'

                break  # found enough SRVs, cannot find more (except for this small %...)

        if verbose:
            print 'debug: synergistic_information: number of SRVs:', len(xrange(len(self), len(pdf_with_srvs)))
            print 'debug: synergistic_information: entropy of SRVs:', \
                pdf_with_srvs.entropy(range(len(self), len(pdf_with_srvs)))
            print 'debug: synergistic_information: H(SRV_i) =', \
                [pdf_with_srvs.entropy([srv_id]) for srv_id in xrange(len(self), len(pdf_with_srvs))]
            print 'debug: synergistic_information: I(Y, SRV_i) =', \
                [pdf_with_srvs.mutual_information(variables_Y, [srv_id])
                        for srv_id in xrange(len(self), len(pdf_with_srvs))]

        syn_info = sum([pdf_with_srvs.mutual_information(variables_Y, [srv_id])
                        for srv_id in xrange(len(self), len(pdf_with_srvs))])

        return syn_info


    def synergistic_entropy(self, variables_X, tolerance_nonsyn_mi=0.05, verbose=True):

        assert False, 'todo: change from information to entropy here, just return different value and do not ask for Y'

        pdf_with_srvs = self.copy()

        syn_entropy = self.synergistic_entropy_upper_bound(variables_X)
        max_ent_per_var = np.log2(self.numvalues)
        max_num_srv_add_trials = int(round(syn_entropy / max_ent_per_var * 2))

        # note: currently I constrain to add SRVs which consist each of only 1 variable each. I guess in most cases
        # this will work fine, but I do not know for sure that this will always be sufficient (i.e., does there
        # ever an MSRV exist with bigger entropy, and if so, is this MSRV not decomposable into parts such that
        # this approach here still works?)

        for i in xrange(max_num_srv_add_trials):
            pdf_with_srvs.append_synergistic_variables(1, initial_guess_summed_modulo=False,
                                                       subject_variables=variables_X, num_repeats=3,
                                                       agnostic_about=range(len(variables_Y) + len(variables_X),
                                                                            len(pdf_with_srvs)))

            # todo: can save one MI calculation here
            new_syn_info = self.synergistic_information_naive([-1], variables_X)
            total_mi = self.mutual_information([-1], variables_X)

            if new_syn_info < syn_entropy * 0.01:
                if (total_mi - new_syn_info) / total_mi > tolerance_nonsyn_mi:  # too much non-synergistic information?
                    pdf_with_srvs.marginalize_distribution(range(len(pdf_with_srvs) - 1))  # remove last srv

                    if verbose:
                        print 'debug: synergistic_information: a last SRV was found but with too much non-syn. info.'
                else:
                    if verbose:
                        print 'debug: synergistic_information: a last (small) ' \
                              'SRV was found, a good one, and now I will stop.'

                break  # assume that no more better SRVs can be found from now on, so stop (save time)
            else:
                if i == max_num_srv_add_trials - 1:
                    warnings.warn('synergistic_information: never converged to adding SRV with zero synergy')

                if (total_mi - new_syn_info) / total_mi > tolerance_nonsyn_mi:  # too much non-synergistic information?
                    pdf_with_srvs.marginalize_distribution(range(len(pdf_with_srvs) - 1))  # remove last srv

                    if verbose:
                        print('debug: synergistic_information: an SRV with new_syn_info/total_mi = '
                                  + str(new_syn_info) + ' / ' + str(total_mi) + ' = '
                                  + str((new_syn_info) / total_mi) + ' was found, which will be removed again because'
                                                                     ' it does not meet the tolerance of '
                                  + str(tolerance_nonsyn_mi))
                else:
                    if verbose:
                        print 'debug: synergistic_information: an SRV with new_syn_info/total_mi = ' \
                              + str(new_syn_info) + ' / ' + str(total_mi) + ' = ' \
                                  + str((new_syn_info) / total_mi) + ' was found, which satisfies the tolerance of ' \
                                  + str(tolerance_nonsyn_mi)

        # if verbose:
        #     print 'debug: synergistic_information: number of SRVs:', len(xrange(len(self), len(pdf_with_srvs)))
        #     print 'debug: synergistic_information: entropy of SRVs:', \
        #         pdf_with_srvs.entropy(range(len(self), len(pdf_with_srvs)))
        #     print 'debug: synergistic_information: H(SRV_i) =', \
        #         [pdf_with_srvs.entropy([srv_id]) for srv_id in xrange(len(self), len(pdf_with_srvs))]
        #     print 'debug: synergistic_information: I(Y, SRV_i) =', \
        #         [pdf_with_srvs.mutual_information(variables_Y, [srv_id])
        #                 for srv_id in xrange(len(self), len(pdf_with_srvs))]

        syn_info = sum([pdf_with_srvs.mutual_information(variables_Y, [srv_id])
                        for srv_id in xrange(len(self), len(pdf_with_srvs))])

        return syn_info


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


    # X marks the spot


    # todo: add optional numvalues? so that the synergistic variables can have more possible values than the
    # current variables (then set all probabilities where the original variables exceed their original max to 0)
    # todo: first you should then probably implement a .increase_num_values(num) or so (or only),
    # or let .numvalues be a list instead of a single value
    def append_synergistic_variables(self, num_synergistic_variables, initial_guess_summed_modulo=False, verbose=False,
                                     subject_variables=None, agnostic_about=None, num_repeats=1, minimize_method=None,
                                     tol_nonsyn_mi_frac=0.05, tol_agn_mi_frac=0.05):
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

        if __debug__:  # looking for bug
            assert max(self.matrix2params_incremental()) < 1.0000001, 'param is out of bound: ' \
                                                                      + str(max(self.matrix2params_incremental()))
            assert min(self.matrix2params_incremental()) > -0.0000001, 'param is out of bound: ' \
                                                                      + str(min(self.matrix2params_incremental()))

        parameter_values_before = list(self.matrix2params_incremental())

        # if __debug__:
        #     debug_params_before = copy.deepcopy(parameter_values_before)

        # a pdf with XORs as appended variables (often already MSRV for binary variables), good initial guess?
        pdf_with_srvs = self.copy()
        pdf_with_srvs.append_variables_using_state_transitions_table(
            state_transitions=lambda vals, mv: [int(np.mod(np.sum(vals), mv))]*num_synergistic_variables)

        assert pdf_with_srvs.numvariables == self.numvariables + num_synergistic_variables

        parameter_values_after = pdf_with_srvs.matrix2params_incremental()

        assert len(parameter_values_after) > len(parameter_values_before), 'should be additional free parameters'
        if np.random.random < 0.1:  # reduce slowdown from this
            # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
            # have to optimize the latter part of parameter_values_after
            np.testing.assert_array_almost_equal(parameter_values_before,
                                                 parameter_values_after[:len(parameter_values_before)])

        # this many parameters (each in [0,1]) must be optimized
        num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

        assert num_synergistic_variables == 0 or num_free_parameters > 0

        if initial_guess_summed_modulo:
            # note: this is xor for binary variables
            initial_guess = parameter_values_after[len(parameter_values_before):]
        else:
            initial_guess = np.random.random(num_free_parameters)

        if verbose:
            debug_pdf_with_srvs = pdf_with_srvs.copy()
            debug_pdf_with_srvs.params2matrix_incremental(list(parameter_values_before) + list(initial_guess))

            # store the synergistic information before the optimization procedure (after procedure should be higher...)
            debug_before_syninfo = debug_pdf_with_srvs.synergistic_information_naive(
                variables_SRV=range(self.numvariables, pdf_with_srvs.numvariables),
                variables_X=range(self.numvariables))

        assert len(initial_guess) == num_free_parameters

        pdf_with_srvs_for_optimization = pdf_with_srvs.copy()

        if not subject_variables is None:
            pdf_subjects_syns_only = pdf_with_srvs_for_optimization.marginalize_distribution(
                list(subject_variables) + range(len(pdf_with_srvs) - num_synergistic_variables, len(pdf_with_srvs)))

            pdf_subjects_only = pdf_subjects_syns_only.marginalize_distribution(range(len(subject_variables)))

            if __debug__ and np.random.random() < 0.01:
                debug_pdf_subjects_only = pdf_with_srvs.marginalize_distribution(subject_variables)

                assert debug_pdf_subjects_only == pdf_subjects_only

            num_free_parameters_synonly = len(pdf_subjects_syns_only.matrix2params_incremental()) \
                                          - len(pdf_subjects_only.matrix2params_incremental())

            parameter_values_static = pdf_subjects_only.matrix2params_incremental()

            initial_guess = np.random.random(num_free_parameters_synonly)

            # pdf_subjects_syns_only should be the new object that fitness_func operates on, instead of
            # pdf_with_srvs_for_optimization
        else:
            # already like this, so simple renaming
            pdf_subjects_syns_only = pdf_with_srvs_for_optimization

            parameter_values_static = parameter_values_before

            num_free_parameters_synonly = num_free_parameters

            # subject_variables = range(len(self))

        upper_bound_synergistic_information = self.synergistic_entropy_upper_bound(subject_variables)
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

        # todo: shouldn't the cost terms in this function not be squared for better convergence?
        def cost_func_subjects_only(free_params, parameter_values_before, extra_cost_rel_error=True):
            """
            This cost function searches for a Pr(S,Y,A,X) such that X is synergistic about S (subject_variables) only.
            This fitness function also taxes any correlation between X and A (agnostic_variables), but does not care
            about the relation between X and Y.
            :param free_params:
            :param parameter_values_before:
            :return:
            """
            assert len(free_params) == num_free_parameters_synonly

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

            pdf_subjects_syns_only.params2matrix_incremental(list(parameter_values_before) + list(free_params))

            # this if-block will add cost for the estimated amount of synergy induced by the proposed parameters,
            # and possible also a cost term for the ratio of synergistic versus non-synergistic info as extra
            if not subject_variables is None:
                assert pdf_subjects_syns_only.numvariables == len(subject_variables) + num_synergistic_variables

                # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                if not extra_cost_rel_error:
                    cost = (upper_bound_synergistic_information - pdf_subjects_syns_only.synergistic_information_naive(
                        variables_SRV=range(len(subject_variables), len(pdf_subjects_syns_only)),
                        variables_X=range(len(subject_variables)))) / upper_bound_synergistic_information
                else:
                    tot_mi = pdf_subjects_syns_only.mutual_information(
                        range(len(subject_variables), len(pdf_subjects_syns_only)),
                        range(len(subject_variables)))

                    indiv_mis = [pdf_subjects_syns_only.mutual_information([var],
                                                                           range(len(subject_variables),
                                                                                 len(pdf_subjects_syns_only)))
                                 for var in range(len(subject_variables))]

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
                assert pdf_subjects_syns_only.numvariables == len(self) + num_synergistic_variables

                if not extra_cost_rel_error:
                    # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                    cost = (upper_bound_synergistic_information - pdf_subjects_syns_only.synergistic_information_naive(
                        variables_SRV=range(len(self), len(pdf_subjects_syns_only)),
                        variables_X=range(len(self)))) / upper_bound_synergistic_information
                else:
                    tot_mi = pdf_subjects_syns_only.mutual_information(
                        range(len(self), len(pdf_subjects_syns_only)),
                        range(len(self)))

                    indiv_mis = [pdf_subjects_syns_only.mutual_information([var],
                                                                           range(len(self),
                                                                                 len(pdf_subjects_syns_only)))
                                 for var in range(len(self))]

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

            # this if-block will add a cost term for not being agnostic to given variables, usually (a) previous SRV(s)
            if not agnostic_about is None:
                assert not subject_variables is None, 'how can all variables be subject_variables and you still want' \
                                                      ' to be agnostic about certain (other) variables? (if you did' \
                                                      ' not specify subject_variables, do so.)'

                # make a conditional distribution of the synergistic variables conditioned on the subject variables
                # so that I can easily make a new joint pdf object with them and quantify this extra cost for the
                # agnostic constraint
                cond_pdf_syns_on_subjects = pdf_subjects_syns_only.conditional_probability_distributions(
                    range(len(subject_variables))
                )

                assert type(cond_pdf_syns_on_subjects) == dict \
                       or isinstance(cond_pdf_syns_on_subjects, ConditionalProbabilities)

                pdf_with_srvs_for_agnostic = self.copy()
                pdf_with_srvs_for_agnostic.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                                            subject_variables)

                if np.ndim(agnostic_about) == 1:
                    # note: cost term for agnostic is in [0,1]
                    cost += (pdf_with_srvs_for_agnostic.mutual_information(synergistic_variables, agnostic_about)) \
                            / upper_bound_agnostic_information
                else:
                    assert np.ndim(agnostic_about) == 2, 'expected list of lists, not more... made a mistake?'

                    assert False, 'I don\'t think this case should happen, according to my 2017 paper should be just ' \
                                  'I(X:A) so ndim==1'

                    for agn_i in agnostic_about:
                        # note: total cost term for agnostic is in [0,1]
                        cost += (1.0 / float(len(agnostic_about))) * \
                                pdf_with_srvs_for_agnostic.mutual_information(synergistic_variables, agn_i) \
                                / upper_bound_agnostic_information

            assert np.isscalar(cost)
            assert np.isfinite(cost)

            return float(cost)

        param_vectors_trace = []

        # these options are altered mainly to try to lower the computation time, which is considerable.
        minimize_options = {'ftol': 1e-6}

        if True:
        # if num_repeats == 1:
        #     optres = minimize(cost_func_subjects_only, initial_guess, bounds=[(0.0, 1.0)]*num_free_parameters_synonly,
        #                       callback=(lambda xv: param_vectors_trace.append(list(xv))) if verbose else None,
        #                       args=(parameter_values_static,), method=minimize_method, options=minimize_options)
        # else:
            optres_list = []

            for ix in xrange(num_repeats):
                optres_ix = minimize(cost_func_subjects_only,
                                     np.random.random(num_free_parameters_synonly) if ix > 0 else initial_guess,
                                     bounds=[(0.0, 1.0)]*num_free_parameters_synonly,
                                     callback=(lambda xv: param_vectors_trace.append(list(xv))) if verbose else None,
                                     args=(parameter_values_static,), method=minimize_method, options=minimize_options)

                if verbose:
                    print 'note: finished a repeat. success=' + str(optres_ix.success) + ', cost=' \
                          + str(optres_ix.fun)

                if not tol_nonsyn_mi_frac is None:
                    pdf_subjects_syns_only.params2matrix_incremental(list(parameter_values_static) + list(optres_ix.x))

                    if subject_variables is None:
                        print 'debug: will set subject_variables=%s' % (range(len(self)))
                        subject_variables = range(len(self))

                    assert not pdf_subjects_syns_only is None

                    tot_mi = pdf_subjects_syns_only.mutual_information(
                                range(len(subject_variables), len(pdf_subjects_syns_only)),
                                range(len(subject_variables)))

                    indiv_mis = [pdf_subjects_syns_only.mutual_information([var],
                                                                           range(len(subject_variables),
                                                                                 len(pdf_subjects_syns_only)))
                                 for var in range(len(subject_variables))]

                    if sum(indiv_mis) / float(tot_mi) > tol_nonsyn_mi_frac:
                        if verbose:
                            print 'debug: in iteration %s I found an SRV but with total MI %s and indiv. MIs %s it ' \
                                  'violates the tol_nonsyn_mi_frac=%s' % (ix, tot_mi, indiv_mis, tol_nonsyn_mi_frac)

                        continue  # don't add this to the list of solutions

                if not tol_agn_mi_frac is None and not agnostic_about is None:
                    if len(agnostic_about) > 0:
                        # note: could reuse the one above, saves a bit of computation
                        pdf_subjects_syns_only.params2matrix_incremental(list(parameter_values_static)
                                                                         + list(optres_ix.x))

                        cond_pdf_syns_on_subjects = pdf_subjects_syns_only.conditional_probability_distributions(
                                                        range(len(subject_variables))
                                                    )

                        # I also need the agnostic variables, which are not in pdf_subjects_syns_only, so construct
                        # the would-be final result (the original pdf with the addition of the newly found SRV)
                        pdf_with_srvs = self.copy()
                        pdf_with_srvs.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                                       given_variables=subject_variables)

                        agn_mi = pdf_with_srvs.mutual_information(agnostic_about, range(len(self), len(pdf_with_srvs)))
                        agn_ent = self.entropy(agnostic_about)

                        if agn_mi / agn_ent > tol_agn_mi_frac:
                            if verbose:
                                print 'debug: in iteration %s I found an SRV but with agn_mi=%s and agn_ent=%s it ' \
                                      'violates the tol_agn_mi_frac=%s' % (ix, agn_mi, agn_ent, tol_nonsyn_mi_frac)

                            continue  # don't add this to the list of solutions

                optres_list.append(optres_ix)

            if verbose and __debug__:
                print 'debug: num_repeats=' + str(num_repeats) + ', all cost values were: ' \
                      + str([resi.fun for resi in optres_list])
                print 'debug: successes =', [resi.success for resi in optres_list]

            optres_list = [resi for resi in optres_list if resi.success]  # filter out the unsuccessful optimizations

            if len(optres_list) == 0:
                raise UserWarning('all ' + str(num_repeats) + ' optimizations using minimize() failed...?!')

            costvals = [res.fun for res in optres_list]
            min_cost = min(costvals)
            optres_ix = costvals.index(min_cost)

            assert optres_ix >= 0 and optres_ix < len(optres_list)

            optres = optres_list[optres_ix]

        if subject_variables is None:
            assert len(optres.x) == num_free_parameters
        else:
            assert len(optres.x) == num_free_parameters_synonly

        assert max(optres.x) <= 1.0000001, 'parameter bound significantly violated, ' + str(max(optres.x))
        assert min(optres.x) >= -0.0000001, 'parameter bound significantly violated, ' + str(min(optres.x))

        # todo: reuse the .append_optimized_variables (or so) instead, passing the cost function only? would also
        # validate that function.

        # clip to valid range
        optres.x = [min(max(xi, 0.0), 1.0) for xi in optres.x]

        # optimal_parameters_joint_pdf = list(parameter_values_before) + list(optres.x)
        # pdf_with_srvs.params2matrix_incremental(optimal_parameters_joint_pdf)

        assert len(pdf_subjects_syns_only.matrix2params_incremental()) == len(parameter_values_static) + len(optres.x)

        pdf_subjects_syns_only.params2matrix_incremental(list(parameter_values_static) + list(optres.x))

        if not subject_variables is None:
            cond_pdf_syns_on_subjects = pdf_subjects_syns_only.conditional_probability_distributions(
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
            pdf_with_srvs = pdf_subjects_syns_only  # all variables are subject

        assert pdf_with_srvs.numvariables == self.numvariables + num_synergistic_variables

        if verbose:
            parameter_values_after2 = pdf_with_srvs.matrix2params_incremental()

            assert len(parameter_values_after2) > len(parameter_values_before), 'should be additional free parameters'

            if not 1.0 in parameter_values_after2 and not 0.0 in parameter_values_after2:
                # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
                # have to optimize the latter part of parameter_values_after
                np.testing.assert_array_almost_equal(parameter_values_before,
                                                     parameter_values_after2[:len(parameter_values_before)])
                np.testing.assert_array_almost_equal(parameter_values_after2[len(parameter_values_before):],
                                                     optres.x)
            else:
                # it can happen that some parameters are 'useless' in the sense that they defined conditional
                # probabilities in the case where the prior (that is conditioned upon) has zero probability. The
                # resulting pdf is then always the same, no matter this parameter value. This can only happen if
                # there is a 0 or 1 in the parameter list, (sufficient but not necessary) so skip the test then...
                pass

            # store the synergistic information before the optimization procedure (after procedure should be higher...)
            debug_after_syninfo = pdf_with_srvs.synergistic_information_naive(variables_SRV=range(self.numvariables,
                                                                                         pdf_with_srvs.numvariables),
                                                                              variables_X=range(self.numvariables))

            if verbose:
                print 'debug: append_synergistic_variables: I started from synergistic information =', \
                    debug_before_syninfo, 'at initial guess. After optimization it became', debug_after_syninfo, \
                    '(should be higher). Optimal params:', \
                    parameter_values_after2[len(parameter_values_before):]

        self.duplicate(pdf_with_srvs)


    def susceptibility_local_single(self, var_id, num_output_variables, perturbation_size=0.01, ntrials=25,
                                    impact_measure='abs', nudge_method='fixed', also_return_pdf_after=False,
                                    auto_reorder=True):
        """

        :return: nudges_array, suscs_array
        :rtype: tuple
        """
        nudges = []
        suscs = []
        pdf_after = []  # optional

        # output marginal pdf for assessing impact, possibly
        pdf_output = self[range(len(self) - num_output_variables, len(self))]
        # cond_pdf_out = self.conditional_probability_distributions(range(len(self) - num_output_variables))

        num_input_variables = len(self) - num_output_variables

        if not var_id == num_input_variables - 1 and auto_reorder:
            pdf_c = self.copy()
            other_input_ixs = [i for i in xrange(num_input_variables) if not i == var_id]
            pdf_c.reorder_variables(other_input_ixs + [var_id] + range(num_input_variables, len(self)))

            # make sure the input variable perturbed is the last listed of the inputs so that it does not have
            # any causal effect on the other inputs, which would overestimate the impact on the output potentially
            return pdf_c.susceptibility_local_single(var_id=num_input_variables - 1,
                                                     num_output_variables=num_output_variables,
                                                     perturbation_size=perturbation_size, ntrials=ntrials,
                                                     impact_measure=impact_measure, nudge_method=nudge_method,
                                                     also_return_pdf_after=also_return_pdf_after,
                                                     auto_reorder=auto_reorder)

        for trial in xrange(ntrials):
            pdf = self.copy()
            nudge = pdf.nudge([var_id], range(len(self) - num_output_variables, len(self)), method=nudge_method,
                              epsilon=perturbation_size)

            if impact_measure in ('sq',):
                pdf_out_nudged = pdf[range(len(self) - num_output_variables, len(self))]

                impact = np.sum(np.power(np.subtract(pdf_output.joint_probabilities,
                                                     pdf_out_nudged.joint_probabilities), 2))
            elif impact_measure in ('abs',):
                pdf_out_nudged = pdf[range(len(self) - num_output_variables, len(self))]

                impact = np.sum(np.abs(np.subtract(pdf_output.joint_probabilities,
                                                     pdf_out_nudged.joint_probabilities)))
            elif impact_measure in ('prob',):
                pdf_out_nudged = pdf[range(len(self) - num_output_variables, len(self))]

                impact = np.subtract(pdf_output.joint_probabilities, pdf_out_nudged.joint_probabilities)
            elif impact_measure in ('kl', 'kld', 'kldiv'):
                pdf_out_nudged = pdf[range(len(self) - num_output_variables, len(self))]

                impact = pdf_output.kullback_leibler_divergence(pdf_out_nudged)
            elif impact_measure in ('hell', 'h', 'hellinger'):
                pdf_out_nudged = pdf[range(len(self) - num_output_variables, len(self))]

                impact = pdf_output.hellinger_distance(pdf_out_nudged)
            else:
                raise NotImplementedError('impact_measure=%s is unknown' % impact_measure)

            suscs.append(impact)
            nudges.append(nudge)
            if also_return_pdf_after:
                pdf_after.append(pdf.copy())

        if not also_return_pdf_after:
            return nudges, suscs
        else:
            return nudges, suscs, pdf_after


    def susceptibilities_local(self, num_output_variables, perturbation_size=0.1, ntrials=25, impact_measure='abs',
                               auto_reorder=True):
        return [self.susceptibility_local_single(varid, num_output_variables, perturbation_size=perturbation_size,
                                                 impact_measure=impact_measure, ntrials=ntrials,
                                                 auto_reorder=auto_reorder)
                for varid in xrange(len(self) - num_output_variables)]


    # def susceptibility_local(self, num_output_variables, perturbation_size=0.1, ntrials=25, auto_reorder=True):
    #     return np.mean([self.susceptibility_local_single(varid, num_output_variables,
    #                                                      perturbation_size=perturbation_size,
    #                                                      ntrials=ntrials, auto_reorder=auto_reorder)
    #                     for varid in xrange(len(self) - num_output_variables)])


    def susceptibility_global(self, num_output_variables, perturbation_size=0.1, ntrials=25,
                              impact_measure='hellinger'):
        """
        Perturb the current pdf Pr(X,Y) by changing Pr(X) slightly to Pr(X'), but keeping Pr(Y|X) fixed. Then
        measure the relative change in mutual information I(X:Y). Do this by Monte Carlo using <ntrials> repeats.
        :param num_output_variables: the number of variables at the end of the list of variables which are considered
        to be Y. The first variables are taken to be X. If your Y is mixed with the X, first do reordering.
        :param perturbation_size:
        :param ntrials:
        :return: expected relative change of mutual information I(X:Y) after perturbation
        :rtype: float
        """
        num_input_variables = len(self) - num_output_variables

        assert num_input_variables > 0, 'makes no sense to compute resilience with an empty set'

        original_mi = self.mutual_information(range(num_input_variables),
                                              range(num_input_variables, num_input_variables + num_output_variables))

        pdf_input_only = self.marginalize_distribution(range(num_input_variables))

        affected_params = pdf_input_only.matrix2params_incremental()  # perturb Pr(X) to Pr(X')
        static_params = self.matrix2params_incremental()[len(affected_params):]  # keep Pr(Y|X) fixed

        pdf_perturbed = self.copy()  # will be used to store the perturbed Pr(X)Pr(Y'|X)

        def clip_to_unit_line(num):  # helper function, make sure all probabilities remain valid
            return max(min(num, 1), 0)

        susceptibilities = []

        pdf_output_only = self[range(num_input_variables, len(self))]

        for i in xrange(ntrials):
            perturbation = np.random.random(len(affected_params))
            perturbation = perturbation / np.linalg.norm(perturbation) * perturbation_size  # normalize vector norm

            pdf_perturbed.params2matrix_incremental(map(clip_to_unit_line, np.add(affected_params, perturbation))
                                                    + static_params)

            # susceptibility = pdf_perturbed.mutual_information(range(num_input_variables),
            #                                                   range(num_input_variables,
            #                                                         num_input_variables + num_output_variables)) \
            #                  - original_mi

            pdf_perturbed_output_only = pdf_perturbed[range(num_input_variables, len(self))]

            if impact_measure in ('kl', 'kld', 'kldiv'):
                susceptibility = pdf_output_only.kullback_leibler_divergence(pdf_perturbed_output_only)
            elif impact_measure in ('hell', 'h', 'hellinger'):
                susceptibility = pdf_output_only.hellinger_distance(pdf_perturbed_output_only)
            else:
                raise NotImplementedError('unknown impact_measure=%s' % impact_measure)

            susceptibilities.append(abs(susceptibility))

        return np.mean(susceptibilities) / original_mi


    def susceptibility_non_local(self, output_variables, variables_X1, variables_X2,
                                 perturbation_size=0.1, ntrials=25):

        if list(variables_X1) + list(variables_X2) + list(output_variables) == range(max(output_variables)+1):
            # all variables are already nice ordered, so call directly susceptibility_non_local_ordered, maybe
            # only extraneous variables need to be deleted

            if len(self) > max(output_variables) + 1:
                pdf_lean = self[:(max(output_variables) + 1)]

                # should be exactly the same function call as in 'else', but on different pdf (removing extraneous
                # variables)
                return pdf_lean.susceptibility_non_local_ordered(len(output_variables),
                                                         num_second_input_variables=len(variables_X2),
                                                         perturbation_size=perturbation_size, ntrials=ntrials)
            else:
                return self.susceptibility_non_local_ordered(len(output_variables),
                                                     num_second_input_variables=len(variables_X2),
                                                     perturbation_size=perturbation_size, ntrials=ntrials)
        else:
            # first reorder, then remove extraneous variables, and then call susceptibility_non_local_ordered

            reordering_relevant = list(variables_X1) + list(variables_X2) + list(output_variables)

            extraneous_vars = np.setdiff1d(range(len(self)), reordering_relevant)

            reordering = reordering_relevant + list(extraneous_vars)

            pdf_reordered = self.copy()
            pdf_reordered.reorder_variables(reordering)

            pdf_reordered = pdf_reordered[:len(reordering_relevant)]  # keep only relevant variables (do in reorder?)

            return pdf_reordered.susceptibility_non_local_ordered(len(output_variables),
                                                                  num_second_input_variables=len(variables_X2),
                                                                  perturbation_size=perturbation_size, ntrials=ntrials)


    def kullback_leibler_divergence(self, other_pdf):

        div = 0.0

        assert len(other_pdf) == len(self), 'must have same number of variables (marginalize first)'

        for states in self.statespace():
            assert len(states) == len(other_pdf)

            if other_pdf(states) > 0:
                div += self(states) * np.log2(self(states) / other_pdf(states))
            else:
                if self(states) > 0:
                    div += 0.0
                else:
                    div = np.inf  # divergence becomes infinity if the support of Q is not that of P

        return div


    def hellinger_distance(self, other_pdf):

        div = 0.0

        assert len(other_pdf) == len(self), 'must have same number of variables (marginalize first)'

        for states in self.statespace():
            assert len(states) == len(other_pdf)

            div += np.power(np.sqrt(self(states)) - np.sqrt(other_pdf(states)), 2)

        return 1/np.sqrt(2) * np.sqrt(div)


    def susceptibility_non_local_ordered(self, num_output_variables, num_second_input_variables=1,
                                 perturbation_size=0.1, ntrials=25):
        """
        Perturb the current pdf Pr(X1,X2,Y) by changing Pr(X2|X1) slightly, but keeping Pr(X1) and Pr(X2) fixed. Then
        measure the relative change in mutual information I(X:Y). Do this by Monte Carlo using <ntrials> repeats.
        :param num_second_input_variables: number of variables making up X2.
        :param num_output_variables: the number of variables at the end of the list of variables which are considered
        to be Y. The first variables are taken to be X. If your Y is mixed with the X, first do reordering.
        :param perturbation_size:
        :param ntrials:
        :return: expected relative change of mutual information I(X:Y) after perturbation
        :rtype: float
        """
        num_input_variables = len(self) - num_output_variables

        assert num_input_variables >= 2, 'how can I do non-local perturbation if only 1 input variable?'

        assert num_input_variables > 0, 'makes no sense to compute resilience with an empty set'

        original_mi = self.mutual_information(range(num_input_variables),
                                              range(num_input_variables, num_input_variables + num_output_variables))

        susceptibilities = []

        pdf_inputs = self[range(num_input_variables)]

        cond_pdf_output = self - pdf_inputs

        pdf_outputs = self[range(num_input_variables, len(self))]  # marginal Y

        for i in xrange(ntrials):
            # perturb only among the input variables
            resp = pdf_inputs.perturb_non_local(num_second_input_variables, perturbation_size)

            pdf_perturbed = resp.pdf + cond_pdf_output

            pdf_perturbed_outputs = pdf_perturbed[range(num_input_variables, len(self))]  # marginal Y'

            # susceptibility = pdf_perturbed.mutual_information(range(num_input_variables),
            #                                                   range(num_input_variables,
            #                                                         num_input_variables + num_output_variables)) \
            #                  - original_mi

            # TODO: this compares the marginals of the outputs, but the MI between the old and new output which
            # would be better, but is more involved.
            susceptibility = pdf_outputs.kullback_leibler_divergence(pdf_perturbed_outputs)

            susceptibilities.append(abs(susceptibility))

        return np.mean(susceptibilities) / original_mi


    class PerturbNonLocalResponse(object):
        pdf = None  # JointProbabilityMatrix object
        # cost_same_output_marginal = None  # float, should be as close to zero as possible.
        # cost_different_relation = None  # float, should be as close to zero as possible
        perturb_size = None  # norm of vector added to params


    def susceptibility(self, variables_Y, variables_X='all', perturbation_size=0.1, only_non_local=False,
                       impact_measure='midiff'):

        # note: at the moment I only perturb per individual variable in X, not jointly; need to supprt this in
        # self.perturb()

        """
        :param impact_measure: 'midiff' for nudge control impact, 'relative' for normalized
        :type impact_measure: str
        """
        if variables_X in ('all', 'auto'):
            variables_X = list(np.setdiff1d(range(len(self)), variables_Y))

        assert len(variables_X) > 0, 'makes no sense to measure susceptibility to zero variables (X)'

        if impact_measure == 'relative':
            pdf_X = self[variables_X]  # marginalize X out of Pr(X,Y)
            cond_pdf_Y = self.conditional_probability_distributions(variables_X)

            if __debug__:
                ent_X = pdf_X.entropy()

            pdf_XX = pdf_X.copy()
            pdf_XX.append_variables_using_state_transitions_table(lambda x, nv: x)  # duplicate X

            if __debug__:
                # duplicating X should not increase entropy in any way
                np.testing.assert_almost_equal(ent_X, pdf_XX.entropy())

            for x2id in xrange(len(variables_X), 2*len(variables_X)):
                pdf_XX.perturb([x2id], perturbation_size=perturbation_size, only_non_local=only_non_local)

            if __debug__:
                # I just go ahead and assume that now the entropy must have increased, since I have exerted some
                # external (noisy) influence
                # note: hitting this assert is highly unlikely but could still happen, namely if the perturbation(s)
                # fail to happen
                # note: could happen if some parameter(s) are 0 or 1, because then there are other parameter values which
                # are irrelevant and have no effect, so those could be changed to satisfy the 'perturbation' constraint
                if not 0 in pdf_XX.matrix2params_incremental() and not 1 in pdf_XX.matrix2params_incremental():
                    assert ent_X != pdf_XX.entropy(), 'ent_X=' + str(ent_X) + ', pdf_XX.entropy()=' + str(pdf_XX.entropy())

            pdf_XXYY = pdf_XX.copy()
            pdf_XXYY.append_variables_using_conditional_distributions(cond_pdf_Y, range(len(variables_X)))
            pdf_XXYY.append_variables_using_conditional_distributions(cond_pdf_Y, range(len(variables_X), 2*len(variables_X)))


            ent_Y = pdf_XXYY.entropy(range(2*len(variables_X), 2*len(variables_X) + len(variables_Y)))
            mi_YY = pdf_XXYY.mutual_information(range(2*len(variables_X), 2*len(variables_X) + len(variables_Y)),
                                                range(2*len(variables_X) + len(variables_Y),
                                                      2*len(variables_X) + 2*len(variables_Y)))

            impact = 1.0 - mi_YY / ent_Y
        elif impact_measure in ('midiff',):
            pdf2 = self.copy()

            pdf2.perturb(variables_X, perturbation_size=perturbation_size, only_non_local=only_non_local)

            impact = np.nan  # pacify IDE
            assert False, 'todo: finish implementing, calc MI in the two cases and return diff'

        return impact


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
        xix = range(perturbed)
        yix = range(perturbed, perturbed + 1)
        zix = range(perturbed + 1, self.numvariables)

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
        condprobs = np.array([[pY_X[xi](yi) for xi in pX.statespace()] for yi in pY.statespace()], dtype=np.float128)

        # np.testing.assert_array_almost_equal(np.sum(condprobs, axis=0), np.ones(nY))

        # note: in total, sum_xi condprobs[yi][xi] must change by amount epsilon[yi], but we
        # have to divide this change into |Y| subterms which sum to epsilon[yi]...

        pXprobs = np.array(pX.joint_probabilities.joint_probabilities, dtype=np.float128)

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

        for j in xrange(ntrials_norm):
            # generate a nudge vector
            epsilon = pY.generate_nudge(eps_norm, np.shape(pY.joint_probabilities.joint_probabilities))

            # print 'debug: epsilon: %s -- norm: %s' % (epsilon, np.linalg.norm(epsilon))

            # clip the nudge vector to let all probabilities stay within bounds
            # epsilon = np.max([epsilon, -pY.joint_probabilities.joint_probabilities], axis=0)
            # epsilon = np.min([epsilon, 1.0 - pY.joint_probabilities.joint_probabilities], axis=0)
            epsilon = np.array(epsilon, dtype=np.float128)  # try to reduce roundoff errors below

            # clip the nudge vector to stay within probabilities [0,1]
            # TODO: generate directly a nudge vector within these bounds? in the helper function?
            for i in xrange(ntrials_clip):  # try to clip the nudge vector within a number of trials
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


    # WARNING: nudge_single() is now preferred!
    # todo: remove this and use only nudge_single()
    def nudge(self, perturbed_variables, output_variables, epsilon=0.01, marginalize_prior_ignored=False,
              method='fixed', nudge=None, verbose=True):
        """

        :param perturbed_variables:
        :param output_variables:
        :param epsilon:
        :param marginalize_prior_ignored:
        :param method:
            'random': a nudge is drawn uniformly random from the hypercube [-epsilon, +epsilon]^|P| or whatever
            subspace of that cube that still leaves all probabilities in the range [0,1]
            'fixed': a nudge is a random vector with norm epsilon, whereas for 'random' the norm can be (much) smaller
        :return:
        """
        perturbed_variables = list(sorted(set(perturbed_variables)))
        output_variables = list(sorted(set(output_variables)))
        ignored_variables = list(np.setdiff1d(range(len(self)), perturbed_variables + output_variables))

        # caution: trial of a new way to do a nudge if there are causal predecessors to the perturbed variables,
        # namely: first marginalize out the perturbed variables, do the nudge, and then glue back together by
        # slightly changing the conditional
        # if min(perturbed_variables) > 0:


        # this performs a nudge on each conditional pdf of the perturbed variables, given the values for variables
        # which appear on the lefthand side of them (these are causal predecessors, which should not be affected by
        # the nudge)
        if min(perturbed_variables + output_variables) > 0 and marginalize_prior_ignored:
            # these variables appear before all perturbed and output variables so there should be no causal effect on
            # them (marginal probabilities the same, however the conditional pdf with the perturbed variables
            # may need to change to make the nudge happen, so here I do that (since nudge() by itself tries to fixate
            # both marginals and conditionals of prior variables, which is not always possible (as e.g. for a copied variable)
            ignored_prior_variables = range(min(perturbed_variables + output_variables))

            pdf_prior = self[ignored_prior_variables]  # marginal of prior variables

            # conditional pdfs of the rest
            cond_pdfs = self.conditional_probability_distributions(ignored_prior_variables)

            # the pdf's in the conditional cond_pdfs now are missing the first len(ignored_prior_variables)
            new_pv = np.subtract(perturbed_variables, len(ignored_prior_variables))
            new_ov = np.subtract(output_variables, len(ignored_prior_variables))
            new_iv = np.subtract(ignored_variables, len(ignored_prior_variables))

            nudge_dict = dict()

            for k in cond_pdfs.iterkeys():
                # do an independent and random nudge for each marginal pdf given the values for the prior variables...
                # for each individual instance this may create a correlation with the prior variables, but on average
                # (or for large num_values) it will not.
                nudge_dict[k] = cond_pdfs[k].nudge(perturbed_variables=new_pv, output_variables=new_ov, epsilon=epsilon)

            new_pdf = pdf_prior + cond_pdfs

            self.duplicate(new_pdf)

            return nudge_dict
        # code assumes
        # if max(perturbed_variables) == len(perturbed_variables) - 1 \
        #         and max(output_variables) == len(perturbed_variables) + len(output_variables) - 1:
        # todo: doing this rearrangement seems wrong? because the causal predecessors should be first in the list
        # of variables, so there should be no causation from a 'later' variable back to an 'earlier' variable?
        # elif max(perturbed_variables) == len(perturbed_variables) - 1:
        elif True:  # see if this piece of code can also handle ignored variables to precede the perturbed variables
            pdf_P = self[perturbed_variables]  # marginalize X out of Pr(X,Y)
            pdf_I = self[ignored_variables]
            pdf_IO_P = self.conditional_probability_distributions(perturbed_variables)
            pdf_P_I = self.conditional_probability_distributions(ignored_variables,
                                                                 object_variables=perturbed_variables)
            pdf_O_IP = self.conditional_probability_distributions(ignored_variables + perturbed_variables,
                                                                 object_variables=output_variables)

            # initial limits, to be refined below
            max_nudge = np.ones(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * epsilon
            min_nudge = -np.ones(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * epsilon

            for sx in pdf_P.statespace():
                for sy in pdf_IO_P.itervalues().next().statespace():
                    if pdf_IO_P[sx](sy) != 0.0:
                        # follows from equality (P(p) + nudge(p)) * P(rest|p) == 1
                        # max_nudge[sx] = min(max_nudge[sx], 1.0 / pdf_IO_P[sx](sy) - pdf_P(sx))
                        # I think this constraint is always superseding the above one, so just use this:
                        max_nudge[sx] = min(max_nudge[sx], 1.0 - pdf_P(sx))
                    else:
                        pass  # this pair of sx and sy is impossible so no need to add a constraint for it?
                # note: I removed the division because in the second line it adds nothing
                # min_nudge[sx + sy] = 0.0 / pdf_IO_P[sx](sy) - pdf_P(sx)
                min_nudge[sx] = max(min_nudge[sx], 0.0 - pdf_P(sx))

                # same nudge will be added to each P(p|i) so must not go out of bounds for any pdf_P_I[si]
                for pdf_P_i in pdf_P_I.itervalues():
                    max_nudge[sx] = min(max_nudge[sx], 1.0 - pdf_P_i(sx))
                    min_nudge[sx] = max(min_nudge[sx], 0.0 - pdf_P_i(sx))

                # I think this should not happen. Worst case, max_nudge[sx + sy] == min_nudge[sx + sy], like
                # when P(p)=1 for I=1 and P(p)=0 for I=1, then only a nudge of 0 could be added to P(b).
                # small round off error?
                # alternative: consider making the nudge dependent on I, which gives more freedom but I am not sure
                # if then e.g. the expected correlation is still zero. (Should be, right?)
                assert max_nudge[sx] >= min_nudge[sx], 'unsatisfiable conditions for additive nudge!'

                # note: this is simply a consequence of saying that a nudge should change only a variable's own
                # probabilities, not the conditional probabilities of this variable given other variables
                # mechanism)
                # NOTE: although... it seems inevitable that conditional probabilities change?
                # NOTE: well, in the derivation you assume that the connditional p(B|A) does not change -- at the moment
                # (try it?)
                if max_nudge[sx] == min_nudge[sx]:
                    warnings.warn('max_nudge[sx] == min_nudge[sx], meaning that I cannot find a single nudge'
                                  ' \epsilon_a for all ' + str(len(list(pdf_P_I.itervalues())))
                                  + ' pdf_P_i of pdf_P_I=P(perturbed_variables | ignored_variables='
                                  + str(ignored_variables) + '); sx=' + str(sx))
                    # NOTE: implement a different nudge for each value for ignored_variables? Then this does not happen
                    # NOTE: then the nudge may correlate with ignored_variables, which is not what you want because
                    # then you could be adding correlations?

            range_nudge = max_nudge - min_nudge

            max_num_trials_nudge_gen = 10000
            max_secs = 20.
            time_before = time.time()
            if nudge is None:
                for trial in xrange(max_num_trials_nudge_gen):
                    if method == 'random' or method == 'fixed':
                        nudge = np.random.random(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * range_nudge \
                                + min_nudge

                        # sum_nudge = np.sum(nudge)  # should become 0, but currently will not be
                        # correction = np.ones(np.shape(pdf_P.joint_probabilities.joint_probabilities)) / nudge.size * sum_nudge
                        # nudge -= correction
                        nudge -= np.mean(nudge)  # make sum up to 0

                        if method == 'fixed':  # make sure the norm is right
                            nudge *= epsilon / np.linalg.norm(nudge)

                        if np.all(nudge <= max_nudge) and np.all(nudge >= min_nudge):
                            break  # found a good nudge!
                        else:
                            if trial == max_num_trials_nudge_gen - 1:
                                raise UserWarning('max_num_trials_nudge_gen=' + str(max_num_trials_nudge_gen)
                                                  + ' was not enough to find a good nudge vector. '
                                                    'max_nudge=%s (norm: %s), min_nudge=%s'
                                                  % (max_nudge, np.linalg.norm(max_nudge), min_nudge))
                            elif time.time() - time_before > max_secs:
                                raise UserWarning('max_secs=%s was not enough to find a good nudge vector. '
                                                  'trial=%s out of %s' % (max_secs, trial, max_num_trials_nudge_gen))
                            else:
                                continue  # darn, let's try again
                    else:
                        raise NotImplementedError('unknown method: %s' % method)
                # todo: if the loop above fails then maybe do a minimize() attempt?
            else:
                nudge = np.array(nudge)

                if np.all(nudge <= max_nudge) and np.all(nudge >= min_nudge):
                    pass  # done
                else:
                    orig_nudge = copy.deepcopy(nudge)

                    def cost(nudge):
                        overshoot = np.sum(np.max([nudge - max_nudge, np.zeros(nudge.shape)], axis=0))
                        undershoot = np.sum(np.max([min_nudge - nudge, np.zeros(nudge.shape)], axis=0))
                        # these two are actually not needed I think I was thinking of probabilities when I typed it:
                        overbound = np.sum(np.max([nudge - np.ones(nudge.shape), np.zeros(nudge.shape)], axis=0))
                        underbound = np.sum(np.max([np.zeros(nudge.shape) - nudge, np.zeros(nudge.shape)], axis=0))

                        dist = np.linalg.norm(nudge - orig_nudge)

                        return np.sum(np.power([overshoot, undershoot, overbound, underbound], 2)) \
                               + np.power(dist, 2) \
                               + np.power(np.linalg.norm(nudge) - epsilon, 1.0)  # try to get a nudge of desired norm

                    optres = minimize(cost, nudge)

                    max_num_minimize = 5
                    trial = 0
                    while not optres.success and trial < max_num_minimize:
                        trial += 1
                        optres = minimize(cost, nudge)

                    nudge = optres.x

                    if not optres.success:
                        raise UserWarning('You specified nudge=%s but it would make certain probabilities out of '
                                          'bounds (0,1). So I tried a minimize() %s times but it failed.'
                                          '\nmin_nudge=%s'
                                          '\nmax_nudge=%s' % (orig_nudge, max_num_minimize, min_nudge, max_nudge))
                    elif not np.all(nudge <= max_nudge) and np.all(nudge >= min_nudge):
                        raise UserWarning('You specified nudge=%s but it would make certain probabilities out of '
                                          'bounds (0,1). So I tried a minimize() step but it found a nudge still out '
                                          'of the allowed min and max nudge. cost=%s. I tried %s times.'
                                          '\nmin_nudge=%s'
                                          '\nmax_nudge=%s' % (orig_nudge, optres.fun, max_num_minimize,
                                                              min_nudge, max_nudge))
                    elif verbose:
                        print 'debug: you specified nudge=%s but it would make certain probabilities out of ' \
                              'bounds (0,1). So I tried a minimize() step and I got nudge=%s (norm %s, cost %s)' \
                              % (nudge, np.linalg.norm(nudge), optres.fun)

            np.testing.assert_almost_equal(np.sum(nudge), 0.0, decimal=10)  # todo: remove after a while

            # this is the point of self.type_prob ...
            assert type(pdf_P.joint_probabilities.joint_probabilities.flat.next()) == self.type_prob

            # if len(ignored_variables) > 0:  # can this become the main code, so not an if? (also works if
            if True:
                # len(ignored*) == 0?)
                # new attempt for making the new joint pdf
                new_joint_probs = -np.ones(np.shape(self.joint_probabilities.joint_probabilities))

                # print 'debug: ignored_variables = %s, len(self) = %s' % (ignored_variables, len(self))

                for s in self.statespace():
                    # shorthands: the states pertaining to the three subsets of variables
                    sp = tuple(np.take(s, perturbed_variables))
                    so = tuple(np.take(s, output_variables))
                    si = tuple(np.take(s, ignored_variables))

                    new_joint_probs[s] = pdf_I(si) * min(max((pdf_P_I[si](sp) + nudge[sp]), 0.0), 1.0) * pdf_O_IP[si + sp](so)

                    # might go over by a tiny bit due to roundoff, then just clip
                    if -1e-10 < new_joint_probs[s] < 0:
                        new_joint_probs[s] = 0
                    elif 1 < new_joint_probs[s] < 1 + 1e-10:
                        new_joint_probs[s] = 1

                    assert 0 <= new_joint_probs[s] <= 1, 'new_joint_probs[s] = ' + str(new_joint_probs[s])

                self.reset(self.numvariables, self.numvalues, new_joint_probs)
            else:
                print 'debug: perturbed_variables = %s, len(self) = %s' % (perturbed_variables, len(self))

                new_probs = pdf_P.joint_probabilities.joint_probabilities + nudge

                # this is the point of self.type_prob ...
                assert type(new_probs.flat.next()) == self.type_prob

                # # todo: remove test once it works a while
                # if True:
                #     assert np.max(new_probs) <= 1.0, 'no prob should be >1.0: ' + str(np.max(new_probs))
                #     assert np.min(new_probs) >= 0.0, 'no prob should be <0.0: ' + str(np.min(new_probs))

                # assert np.max(nudge) <= abs(epsilon), 'no nudge should be >' + str(epsilon) + ': ' + str(np.max(nudge))
                # assert np.min(nudge) >= -abs(epsilon), 'no nudge should be <' + str(-epsilon) + ': ' + str(np.min(nudge))

                # total probability mass should be unchanged  (todo: remove test once it works a while)
                np.testing.assert_almost_equal(np.sum(new_probs), 1)

                if __debug__:
                    pdf_X_orig = pdf_P.copy()

                pdf_P.joint_probabilities.reset(new_probs)

                if len(pdf_IO_P) > 0:
                    if len(pdf_IO_P.itervalues().next()) > 0:  # non-zero number of variables in I or O?
                        self.duplicate(pdf_P + pdf_IO_P)
                    else:
                        self.duplicate(pdf_P)
                else:
                    self.duplicate(pdf_P)

            return nudge
        else:
            # reorder the variables such that the to-be-perturbed variables are first, then call the same function
            # again (ending up in the if block above) and then reversing the reordering.

            # output_variables = list(np.setdiff1d(range(len(self)), perturbed_variables))

            # WARNING: this seems incorrect, now there is also causal effect on variables which appear before
            # all perturbed variables, which is not intended as the order in which variables appear give a partial
            # causality predecessor ordering
            # todo: remove this else-clause? Now the above elif-clause can handle everything

            self.reorder_variables(perturbed_variables + ignored_variables + output_variables)

            ret = self.nudge(range(len(perturbed_variables)), output_variables, epsilon=epsilon)

            self.reverse_reordering_variables(perturbed_variables + ignored_variables + output_variables)

            return ret

    # # I adopt the convention that variables are ordered in a causal predecessor partial ordering,
    # # so variable 1 cannot causally influence 0 but 0 *can* causally influence 1. This ordering should always be
    # # possible (otherwise add time to make loops). Ignored variables are marginalized out first
    # # todo: does not work satisfactorily
    # def nudge_new(self, perturbed_variables, output_variables, force_keep=None, epsilon=0.01):
    #
    #     if force_keep is None:
    #         force_keep = []
    #
    #     perturbed_variables = sorted(list(set(perturbed_variables)))
    #     output_variables = sorted(list(set(output_variables)))
    #     ignored_variables = list(np.setdiff1d(range(len(self)), perturbed_variables + output_variables))
    #     ignored_variables = list(np.setdiff1d(ignored_variables, force_keep))
    #
    #     assert len(set(perturbed_variables + output_variables)) == len(perturbed_variables) + len(output_variables), \
    #         'perturbed_variables=%s and output_variables%s should not overlap' % (perturbed_variables, output_variables)
    #
    #     if len(ignored_variables) > 0:
    #         self.duplicate(self.marginalize_distribution(perturbed_variables + output_variables))
    #
    #         perturbed_variables = [pv - np.sum(np.less(ignored_variables, pv)) for pv in perturbed_variables]
    #         output_variables = [ov - np.sum(np.less(ignored_variables, ov)) for ov in output_variables]
    #
    #         assert len(list(np.setdiff1d(range(len(self)), perturbed_variables + output_variables))) == 0, \
    #             'perturbed_variables=%s and output_variables%s should cover all variables now: %s' \
    #             % (perturbed_variables, output_variables,
    #                np.setdiff1d(range(len(self)), perturbed_variables + output_variables))
    #
    #         return self.nudge_new(perturbed_variables=perturbed_variables,
    #                               output_variables=output_variables,
    #                               epsilon=epsilon)
    #
    #     # code assumes
    #     # if max(perturbed_variables) == len(perturbed_variables) - 1 \
    #     #         and max(output_variables) == len(perturbed_variables) + len(output_variables) - 1:
    #     # todo: doing this rearrangement seems wrong? because the causal predecessors should be first in the list
    #     # of variables, so there should be no causation from a 'later' variable back to an 'earlier' variable?
    #     # if max(perturbed_variables) == len(perturbed_variables) - 1:
    #     if True:  # test
    #         pdf_P = self[perturbed_variables]  # marginalize X out of Pr(X,Y)
    #         pdf_I = self[ignored_variables]
    #         pdf_IO_P = self.conditional_probability_distributions(perturbed_variables)
    #         pdf_P_I = self.conditional_probability_distributions(ignored_variables,
    #                                                              object_variables=perturbed_variables)
    #         pdf_O_IP = self.conditional_probability_distributions(ignored_variables + perturbed_variables,
    #                                                               object_variables=output_variables)
    #
    #         # initial limits, to be refined below
    #         max_nudge = np.ones(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * epsilon
    #         min_nudge = -np.ones(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * epsilon
    #
    #         for sx in pdf_P.statespace():
    #             for sy in pdf_IO_P.itervalues().next().statespace():
    #                 # follows from (P(p) + nudge(p)) * P(rest|p) == 1
    #                 max_nudge[sx] = min(max_nudge[sx], 1.0 / pdf_IO_P[sx](sy) - pdf_P(sx))
    #             # note: I removed the division because in the second line it adds nothing
    #             # min_nudge[sx + sy] = 0.0 / pdf_IO_P[sx](sy) - pdf_P(sx)
    #             min_nudge[sx] = max(min_nudge[sx], 0.0 - pdf_P(sx))
    #
    #             # same nudge will be added to each P(p|i) so must not go out of bounds for any pdf_P_I[si]
    #             for pdf_P_i in pdf_P_I.itervalues():
    #                 max_nudge[sx] = min(max_nudge[sx], 1.0 - pdf_P_i(sx))
    #                 min_nudge[sx] = max(min_nudge[sx], 0.0 - pdf_P_i(sx))
    #
    #             # I think this should not happen. Worst case, max_nudge[sx + sy] == min_nudge[sx + sy], like
    #             # when P(p)=1 for I=1 and P(p)=0 for I=1, then only a nudge of 0 could be added to P(b).
    #             # small round off error?
    #             # alternative: consider making the nudge dependent on I, which gives more freedom but I am not sure
    #             # if then e.g. the expected correlation is still zero. (Should be, right?)
    #             assert max_nudge[sx] >= min_nudge[sx], 'unsatisfiable conditions for additive nudge!'
    #
    #             if max_nudge[sx] == min_nudge[sx]:
    #                 warnings.warn('max_nudge[sx] == min_nudge[sx], meaning that I cannot find a single nudge'
    #                               ' \epsilon_a for all ' + str(len(list(pdf_P_I.itervalues())))
    #                               + ' pdf_P_i of pdf_P_I=P(perturbed_variables | ignored_variables='
    #                               + str(ignored_variables) + ')')
    #                 # todo: implement a different nudge for each value for ignored_variables? Then this does not happen
    #                 # NOTE: then the nudge may correlate with ignored_variables, which is not what you want because
    #                 # then you could be adding correlations?
    #
    #         range_nudge = max_nudge - min_nudge
    #
    #         max_num_trials_nudge_gen = 1000
    #         for trial in xrange(max_num_trials_nudge_gen):
    #             nudge = np.random.random(np.shape(pdf_P.joint_probabilities.joint_probabilities)) * range_nudge \
    #                     + min_nudge
    #
    #             sum_nudge = np.sum(nudge)  # should become 0, but currently will not be
    #
    #             correction = np.ones(
    #                 np.shape(pdf_P.joint_probabilities.joint_probabilities)) / nudge.size * sum_nudge
    #
    #             nudge -= correction
    #
    #             if np.all(nudge <= max_nudge) and np.all(nudge >= min_nudge):
    #                 break  # found a good nudge!
    #             else:
    #                 if trial == max_num_trials_nudge_gen - 1:
    #                     warnings.warn(
    #                         'max_num_trials_nudge_gen=' + str(max_num_trials_nudge_gen) + ' was not enough'
    #                         + ' to find a good nudge vector. Will fail...')
    #
    #                 continue  # darn, let's try again
    #
    #         np.testing.assert_almost_equal(np.sum(nudge), 0.0, decimal=12)
    #
    #         # this is the point of self.type_prob ...
    #         assert type(pdf_P.joint_probabilities.joint_probabilities.flat.next()) == self.type_prob
    #
    #         if len(ignored_variables) > 0:  # can this become the main code, so not an if? (also works if
    #             # len(ignored*) == 0?)
    #             # new attempt for making the new joint pdf
    #             new_joint_probs = -np.ones(np.shape(self.joint_probabilities.joint_probabilities))
    #
    #             for s in self.statespace():
    #                 # shorthands: the states pertaining to the three subsets of variables
    #                 sp = tuple(np.take(s, perturbed_variables))
    #                 so = tuple(np.take(s, output_variables))
    #                 si = tuple(np.take(s, ignored_variables))
    #
    #                 new_joint_probs[s] = pdf_I(si) * (pdf_P_I[si](sp) + nudge[sp]) * pdf_O_IP[si + sp](so)
    #
    #                 # might go over by a tiny bit due to roundoff, then just clip
    #                 if -1e-10 < new_joint_probs[s] < 0:
    #                     new_joint_probs[s] = 0
    #                 elif 1 < new_joint_probs[s] < 1 + 1e-10:
    #                     new_joint_probs[s] = 1
    #
    #                 assert 0 <= new_joint_probs[s] <= 1, 'new_joint_probs[s] = ' + str(new_joint_probs[s])
    #
    #             self.reset(self.numvariables, self.numvalues, new_joint_probs)
    #         else:
    #             new_probs = pdf_P.joint_probabilities.joint_probabilities + nudge
    #
    #             # this is the point of self.type_prob ...
    #             assert type(new_probs.flat.next()) == self.type_prob
    #
    #             # todo: remove test once it works a while
    #             if True:
    #                 assert np.max(new_probs) <= 1.0, 'no prob should be >1.0: ' + str(np.max(new_probs))
    #                 assert np.min(new_probs) >= 0.0, 'no prob should be <0.0: ' + str(np.min(new_probs))
    #
    #             # assert np.max(nudge) <= abs(epsilon), 'no nudge should be >' + str(epsilon) + ': ' + str(np.max(nudge))
    #             # assert np.min(nudge) >= -abs(epsilon), 'no nudge should be <' + str(-epsilon) + ': ' + str(np.min(nudge))
    #
    #             # total probability mass should be unchanged  (todo: remove test once it works a while)
    #             np.testing.assert_almost_equal(np.sum(new_probs), 1)
    #
    #             if __debug__:
    #                 pdf_X_orig = pdf_P.copy()
    #
    #             pdf_P.joint_probabilities.reset(new_probs)
    #
    #             self.duplicate(pdf_P + pdf_IO_P)
    #
    #         return nudge
    #     else:
    #         raise UserWarning('you should not mix perturbed_variables and output_variables. Not sure yet how to'
    #                           ' implement that. The output variables that occur before a perturbed variable should'
    #                           ' probably be swapped but with the condition that they become independent from the'
    #                           ' said perturbed variable(s), however the output variable could also be a causal '
    #                           ' predecessor for that perturbed variable and swapping them means that this'
    #                           ' causal relation would be lost (under the current assumption that variables are'
    #                           ' ordered as causal predecessors of each other). Maybe split this up in different'
    #                           ' nudge scenarios?')


    # helper function
    def logbase(self, x, base, replace_zeros=True):
        """
        A wrapper around np.log(p) which will return 0 if p is 0, which is useful for MI calc. because 0 log 0 = 0
        by custom.
        :type replace_zeros: bool
        """

        if replace_zeros:
            if not np.isscalar(x):
                x2 = copy.deepcopy(x)

                if replace_zeros:
                    np.place(x2, x2 == 0, 1)
            else:
                x2 = x if x != 0 else 1
        else:
            x2 = x

        if base == 2:
            ret = np.log2(x2)
        elif base == np.e:
            ret = np.log(x2)
        else:
            ret = np.log(x2) / np.log(base)

        return ret


    def causal_impact_of_nudge(self, perturbed_variables, output_variables='auto', hidden_variables=None,
                               epsilon=0.01, num_samples=20, base=2):
        """
        This function determines the *direct* causal impact of <perturbed_variables> on <output_variables>. All other
         variables are assumed 'fixed', meaning that the question becomes: if I fix all other variables' values, will
         a change in <perturbed_variables> result in a change in <output_variables>?

         This function will more specifically determine to what extent the mutual information between
         <perturbed_variables> and <output_variables> is 'causal'. If the return object is <impact> then this
         extent (fraction) can be calculated as "impact.avg_impact / (impact.avg_corr - impact.avg_mi_diff)".
        :rtype: CausalImpactResponse
        """

        # todo: add also an unobserved_variables list or so. now ignored_variables are actually considered fixed,
        # not traced out

        if hidden_variables is None:
            hidden_variables = []

        if output_variables in ('auto', 'all'):
            fixed_variables = []

            output_variables = list(np.setdiff1d(range(len(self)), list(perturbed_variables) + list(hidden_variables)))
        else:
            fixed_variables = list(np.setdiff1d(range(len(self)), list(perturbed_variables) + list(hidden_variables)
                                                + list(output_variables)))

        perturbed_variables = sorted(list(set(perturbed_variables)))
        hidden_variables = sorted(list(set(hidden_variables)))
        output_variables = sorted(list(set(output_variables)))
        fixed_variables = sorted(list(set(fixed_variables)))

        # print 'debug: output_variables =', output_variables
        # print 'debug: hidden_variables =', hidden_variables
        # print 'debug: fixed_variables =', fixed_variables
        # print 'debug: perturbed_variables =', perturbed_variables

        # assert sorted(perturbed_variables + hidden_variables + output_variables + fixed_variables)

        if not hidden_variables is None:
            if len(hidden_variables) > 0:
                pdf = self.copy()

                pdf.reorder_variables(list(perturbed_variables) + list(fixed_variables) + list(output_variables)
                                      + list(hidden_variables))

                # sum out the hidden variables
                pdf = pdf.marginalize_distribution(range(len(list(perturbed_variables) + list(fixed_variables)
                                                             + list(output_variables))))

        ret = CausalImpactResponse()

        ret.perturbed_variables = perturbed_variables

        ret.mi_orig = self.mutual_information(perturbed_variables, output_variables, base=base)

        ret.mi_nudged_list = []
        ret.impacts_on_output = []
        ret.correlations = []

        ret.nudges = []
        ret.upper_bounds_impact = []

        pdf_out = self.marginalize_distribution(output_variables)
        pdf_pert = self.marginalize_distribution(perturbed_variables)

        cond_out = self.conditional_probability_distributions(perturbed_variables, output_variables)

        assert len(cond_out.itervalues().next()) == len(output_variables), \
            'len(cond_out.itervalues().next()) = ' + str(len(cond_out.itervalues().next())) \
            + ', len(output_variables) = ' + str(len(output_variables))

        for i in xrange(num_samples):
            pdf = self.copy()

            nudge = pdf.nudge(perturbed_variables, output_variables, epsilon)

            upper_bound_impact = np.sum([np.power(np.sum([nudge[a] * cond_out[a](b)
                                                          for a in pdf_pert.statespace()]), 2) / pdf_out(b)
                                         for b in pdf_out.statespace()])

            # NOTE: below I do this multiplier to make everything work out, but I do not yet understand why, but anyway
            # then I have to do it here as well -- otherwise I cannot compare the below <impact> with this upper bound
            upper_bound_impact *= 1.0 / 2.0 * 1.0 / np.log(base)

            np.testing.assert_almost_equal(np.sum(nudge), 0, decimal=12,
                                           err_msg='more strict: 0 != ' + str(np.sum(nudge)))
            # assert np.sum([nudge[pvs] for pvs in cond_out.iterkeys()]) == 0, \
            #     'more strict to find bug: ' + str(np.sum([nudge[pvs] for pvs in cond_out.iterkeys()]))

            ret.nudges.append(nudge)
            ret.upper_bounds_impact.append(upper_bound_impact)

            ### determine MI difference

            ret.mi_nudged_list.append(pdf.mutual_information(perturbed_variables, output_variables, base=base))

            ### compute causal impact term

            pdf_out_new = pdf.marginalize_distribution(output_variables)

            impact = np.sum([np.power(pdf_out_new(b) - pdf_out(b), 2) / pdf_out(b)
                             for b in pdf_out.statespace()])
            # NOTE: I do not yet understand where the factor 1/2 comes from!
            # (note to self: the upper bound of i_b is easily derived, but for clearly 100% causal relations
            # this <impact> falls  short of the upper bound by exactly this multiplier... just a thought. Maybe
            # I would have to correct <correlations> by this, and not impact?)
            impact *= 1.0/2.0 * 1.0/np.log(base)
            ret.impacts_on_output.append(impact)

            # if __debug__:
            # the last try-catch seems to fail a lot, don't know why, maybe roundoff, but eps=0.01 seems to always
            # work, so go with that.
            if False:
                # for some reason this seems to be more prone to roundoff errors than above... strange...
                debug_impact = np.sum(np.power(pdf_out_new.joint_probabilities - pdf_out.joint_probabilities, 2)
                                      / pdf_out.joint_probabilities.joint_probabilities)

                debug_impact *= 1.0/2.0 * 1.0/np.log2(base)

                if np.random.randint(10) == 0:
                    array1 = np.power(pdf_out_new.joint_probabilities - pdf_out.joint_probabilities, 2) \
                             / pdf_out.joint_probabilities.joint_probabilities
                    array2 = [np.power(pdf_out_new(b) - pdf_out(b), 2) / pdf_out(b)
                              for b in pdf_out.statespace()]

                    # print 'error: len(array1) =', len(array1)
                    # print 'error: len(array2) =', len(array2)

                    np.testing.assert_almost_equal(array1.flatten(), array2)

                # debug_impact seems highly prone to roundoff errors, don't know why, but coupled with the pairwise
                # testing in the if block above and this more messy test I think it is good to know at least that
                # the two are equivalent, so probably correct
                try:
                    np.testing.assert_almost_equal(impact, debug_impact, decimal=3)
                except AssertionError as e:
                    warnings.warn('np.testing.assert_almost_equal(impact=' + str(impact)
                                  + ', debug_impact=' + str(debug_impact) + ', decimal=3): ' + str(e))

            assert impact >= 0.0

            ### determine correlation term (with specific surprise)

            if __debug__:
                # shorthand, but for production mode I am worried that such an extra function call to a very central
                # function (in Python) may slow things down a lot, so I do it in debug only (seems to take 10-25%
                # more time indeed)
                # def logbase_debug(x, base):
                #     if x != 0:
                #         if base==2:
                #             return np.log2(x)
                #         elif base==np.e:
                #             return np.log(x)
                #         else:
                #             return np.log(x) / np.log(base)
                #     else:
                #         # assuming that this log factor is part of a p * log(p) term, and 0 log 0 = 0 by
                #         # common assumption, then the result will now be 0 whereas if -np.inf then it will be np.nan...
                #         return -_finite_inf

                # looking for bug, sometimes summing nudges in this way results in non-zero
                try:
                    np.testing.assert_almost_equal(np.sum([nudge[pvs] for pvs in pdf_pert.statespace()]), 0,
                                                   err_msg='shape nudge = ' + str(np.shape(nudge)))
                except IndexError as e:
                    print 'error: shape nudge = ' + str(np.shape(nudge))

                    raise IndexError(e)
                # assert np.sum([nudge[pvs] for pvs in cond_out.iterkeys()]) == 0, 'more strict to find bug'

                if __debug__:
                    if np.any(pdf_out.joint_probabilities.joint_probabilities == 0):
                        warnings.warn('at least one probability in pdf_out is zero, so will be an infinite term'
                                      ' in the surprise terms since it has log(... / pdf_out)')

                debug_surprise_terms_fast = np.array([np.sum(cond_out[pvs].joint_probabilities.joint_probabilities
                                               * self.logbase(cond_out[pvs].joint_probabilities.joint_probabilities
                                                         / pdf_out.joint_probabilities.joint_probabilities, base))
                                               for pvs in cond_out.iterkeys()])

                debug_surprise_sum_fast = np.sum(debug_surprise_terms_fast)

                pdf_pert_new = pdf.marginalize_distribution(perturbed_variables)
                np.testing.assert_almost_equal(nudge, pdf_pert_new.joint_probabilities - pdf_pert.joint_probabilities)

                debug_surprise_terms = np.array([np.sum([cond_out[pvs](ovs)
                                                         * (self.logbase(cond_out[pvs](ovs)
                                                            / pdf_out(ovs), base)
                                                            if cond_out[pvs](ovs) > 0 else 0)
                                          for ovs in pdf_out.statespace()])
                     for pvs in cond_out.iterkeys()])
                debug_surprise_sum = np.sum(debug_surprise_terms)
                debug_mi_orig = np.sum(
                    [pdf_pert(pvs) * np.sum([cond_out[pvs](ovs)
                                             * (self.logbase(cond_out[pvs](ovs) / pdf_out(ovs), base)
                                                if cond_out[pvs](ovs) > 0 else 0)
                             for ovs in pdf_out.statespace()])
                     for pvs in cond_out.iterkeys()])

                if not np.isnan(debug_surprise_sum_fast):
                    np.testing.assert_almost_equal(debug_surprise_sum, debug_surprise_sum_fast)
                if not np.isnan(debug_mi_orig):
                    np.testing.assert_almost_equal(debug_mi_orig, ret.mi_orig)

                assert np.all(debug_surprise_terms >= -0.000001), \
                    'each specific surprise s(a) should be non-negative right? ' + str(np.min(debug_surprise_terms))
                if not np.isnan(debug_surprise_sum_fast):
                    np.testing.assert_almost_equal(debug_surprise_terms_fast, debug_surprise_terms)
                    assert np.all(debug_surprise_terms_fast >= -0.000001), \
                        'each specific surprise s(a) should be non-negative right? ' + str(np.min(debug_surprise_terms_fast))

            def logdiv(p, q, base=2):
                if q == 0:
                    return 0  # in sum p(q|p) log p(q|p) / p(q) this will go alright and prevents an error for NaN
                elif p == 0:
                    return 0  # same
                else:
                    return np.log(p / q) / np.log(base)

            # todo: precompute cond_out[pvs] * np.log2(cond_out[pvs] / pdf_out) matrix?
            if base == 2:
                # note: this is the explicit form, should check if a more implicit numpy array form (faster) will be
                # equivalent
                method_correlation = 'fast'
                # method_correlation = 'slow'
                if not method_correlation == 'fast' \
                        or len(fixed_variables) > 0 \
                        or len(hidden_variables) > 0:  # else-block is not yet adapted for this case, not sure yet how
                    correlation = np.array(
                        [nudge[pvs] * np.sum([cond_out[pvs](ovs) * logdiv(cond_out[pvs](ovs), pdf_out(ovs), base)
                                 for ovs in pdf_out.statespace()]) for pvs in cond_out.iterkeys()], dtype=_type_prob)

                    assert np.all(np.isfinite(correlation)), \
                        'correlation contains non-finite scalars, like NaN probably...'

                    correlation = np.sum(correlation)

                    assert np.isfinite(correlation)
                else:
                    assert len(fixed_variables) == 0, 'not yet supported by this else block, must tile over ' \
                                                        'ignored_variables as well, but not sure if it matters how...'

                    # allprobs = self.joint_probabilities.joint_probabilities  # shorthand
                    try:
                        allprobs = np.reshape(np.array([cond_out[a].joint_probabilities.joint_probabilities
                                              for a in pdf_pert.statespace()],
                                              dtype=_type_prob),
                                              np.shape(pdf_pert.joint_probabilities.joint_probabilities)
                                              + np.shape(cond_out.itervalues().next().joint_probabilities.joint_probabilities))
                    except ValueError as e:
                        print 'error: np.shape(cond_out.itervalues().next().joint_probabilities.joint_probabilities) =', \
                            np.shape(cond_out.itervalues().next().joint_probabilities.joint_probabilities)
                        print 'error: np.shape(self.joint_probabilities.joint_probabilities) =', np.shape(self.joint_probabilities.joint_probabilities)
                        print 'error: np.shape(cond_out.iterkeys().next().joint_probabilities.joint_probabilities) =', np.shape(cond_out.iterkeys().next().joint_probabilities.joint_probabilities)
                        print 'error: np.shape(cond_out.itervalues().next().joint_probabilities.joint_probabilities) =', np.shape(cond_out.itervalues().next().joint_probabilities.joint_probabilities)

                        raise ValueError(e)
                    pdf_out_tiled = np.tile(pdf_out.joint_probabilities.joint_probabilities,
                                            [pdf_pert.numvalues] * len(pdf_pert) + [1])  # shorthand

                    correlation = np.sum(nudge * np.sum(allprobs * (self.logbase(allprobs, base) - self.logbase(pdf_out_tiled, base)),
                                                        axis=tuple(range(len(perturbed_variables), len(self)))))

                    assert np.isfinite(correlation)

                    # if __debug__:
                    #     if num_samples <= 10:
                    #         print 'debug: correlation =', correlation

                # todo: try to avoid iterating over nudge[pvs] but instead do an implicit numpy operation, because
                # this way I get roundoff errors sometimes

                # correlation = np.sum([nudge[pvs] * cond_out[pvs].joint_probabilities.joint_probabilities
                #               * (np.log2(cond_out[pvs].joint_probabilities.joint_probabilities)
                #               - np.log2(pdf_out.joint_probabilities.joint_probabilities))
                #               for pvs in cond_out.iterkeys()])

                if __debug__ and np.random.randint(10) == 0:
                    correlation2 = np.sum([nudge[pvs] * np.sum(cond_out[pvs].joint_probabilities.joint_probabilities
                                  * (self.logbase(cond_out[pvs].joint_probabilities.joint_probabilities, 2)
                                  - self.logbase(pdf_out.joint_probabilities.joint_probabilities, 2)))
                                  for pvs in cond_out.iterkeys()])

                    if not np.isnan(correlation2):
                        np.testing.assert_almost_equal(correlation, correlation2)
            elif base == np.e:
                correlation = np.sum([nudge[pvs] * np.sum(cond_out[pvs].joint_probabilities.joint_probabilities
                                      * (np.log(cond_out[pvs].joint_probabilities.joint_probabilities)
                                         - np.log(pdf_out.joint_probabilities.joint_probabilities)))
                                      for pvs in cond_out.iterkeys()])
            else:
                correlation = np.sum([nudge[pvs] * np.sum(cond_out[pvs].joint_probabilities.joint_probabilities
                                      * (np.log(cond_out[pvs].joint_probabilities.joint_probabilities)
                                         - np.log(pdf_out.joint_probabilities.joint_probabilities)) / np.log(base))
                                      for pvs in cond_out.iterkeys()])

            if __debug__ and not method_correlation == 'fast':
                debug_corr = np.sum(
                    [nudge[pvs] * np.sum([cond_out[pvs](ovs) * self.logbase(cond_out[pvs](ovs) / pdf_out(ovs), base)
                                          for ovs in pdf_out.statespace()])
                     for pvs in cond_out.iterkeys()])

                # should be two ways of computing the same
                np.testing.assert_almost_equal(debug_corr, correlation)

            assert np.isfinite(correlation), 'correlation should be a finite number'

            ret.correlations.append(correlation)

        ret.avg_mi_diff = np.mean(np.subtract(ret.mi_nudged_list, ret.mi_orig))
        ret.avg_impact = np.mean(ret.impacts_on_output)
        ret.avg_corr = np.mean(ret.correlations)

        ret.std_mi_diff = np.std(np.subtract(ret.mi_nudged_list, ret.mi_orig))
        ret.std_impact = np.std(ret.impacts_on_output)
        ret.std_corr = np.std(ret.correlations)

        ret.mi_diffs = np.subtract(ret.mi_nudged_list, ret.mi_orig)

        ret.mi_nudged_list = np.array(ret.mi_nudged_list)  # makes it easier to subtract and such by caller
        ret.impacts_on_output = np.array(ret.impacts_on_output)
        ret.correlations = np.array(ret.correlations)

        assert np.all(np.isfinite(ret.correlations))
        assert not np.any(np.isnan(ret.correlations))

        ret.residuals = ret.impacts_on_output - (ret.correlations - ret.mi_diffs)
        ret.avg_residual = np.mean(ret.residuals)
        ret.std_residual = np.std(ret.residuals)

        # return [avg, stddev]
        # return np.mean(np.subtract(mi_nudged_list, mi_orig)), np.std(np.subtract(mi_nudged_list, mi_orig))
        return ret


    # todo: rename to perturb(..., only_non_local=False)
    def perturb(self, perturbed_variables, perturbation_size=0.1, only_non_local=False):
        """

        """

        subject_variables = np.setdiff1d(range(len(self)), list(perturbed_variables))

        assert len(subject_variables) + len(perturbed_variables)

        # todo: for the case only_non_local=False this function makes no sense, there is then still an optimization
        # procedure, but I think it is desired to have a function which adds a *random* vector to the parameters,
        # and minimize() is not guaranteed to do that in absence of cost terms (other than going out of unit cube)
        if only_non_local == False:
            warnings.warn('current perturb() makes little sense with only_non_local=False, see todo above in code')

        if max(subject_variables) < min(perturbed_variables):
            # the variables are a contiguous block of subjects and then a block of perturbs. Now we can use the property
            # of self.matrix2params_incremental() in that its parameters are ordered to encode dependencies to
            # lower-numbered variables.

            pdf_subs_only = self[range(len(subject_variables))]

            params_subs_only = list(pdf_subs_only.matrix2params_incremental())

            params_subs_perturbs = list(self.matrix2params_incremental())

            num_static_params = len(params_subs_only)
            num_free_params = len(params_subs_perturbs) - num_static_params

            free_params_orig = list(params_subs_perturbs[num_static_params:])

            # using equation: a^2 + b^2 + c^2 + ... = p^2, where p is norm and a..c are vector elements.
            # I can then write s_a * p^2 + s_b(p^2 - s_a * p^2) + ... = p^2, so that a = sqrt(s_a * p^2), where all
            # s_{} are independent coordinates in range [0,1]
            def from_sphere_coords_to_vec(coords, norm=perturbation_size):
                accounted_norm = np.power(norm, 2)

                vec = []

                for coord in coords:
                    if -0.0001 <= coord <= 1.0001:
                        coord = min(max(coord, 0.0), 1.0)
                    assert 0 <= coord <= 1.0, 'invalid coordinate'

                    if accounted_norm < 0.0:
                        assert accounted_norm > -0.0001, 'accounted_norm dropped below zero, seems not just rounding' \
                                                         ' error: ' + str(accounted_norm)

                    accounted_norm_i = coord * accounted_norm

                    vec.append(np.sqrt(accounted_norm_i))

                    accounted_norm -= accounted_norm_i

                # add last vector element, which simply consumes all remaining 'norm' quantity
                vec.append(np.sqrt(accounted_norm))

                # norm of resulting vector should be <norm>
                np.testing.assert_almost_equal(np.linalg.norm(vec), norm)

                return vec

            pdf_perturbs_only = self[range(len(subject_variables), len(subject_variables) + len(perturbed_variables))]
            params_perturbs_only = list(pdf_perturbs_only.matrix2params_incremental())

            def clip_to_unit_line(num):  # helper function, make sure all probabilities remain valid
                return max(min(num, 1), 0)

            pdf_new = self.copy()

            # note: to find a perturbation vector of <num_free_params> length and of norm <perturbation_size> we
            # now just have to find <num_free_params-1> independent values in range [0,1] and retrieve the
            # perturbation vector by from_sphere_coords_to_vec
            # rel_weight_out_of_bounds: the higher the more adversion to going out of hypercube (and accepting the
            # ensuing necessary clipping, which makes the norm of the perturbation vector less than requested)
            def cost_perturb_vec(sphere_coords, return_cost_list=False, rel_weight_out_of_bounds=10.0):
                new_params = params_subs_perturbs  # old param values
                perturb_vec = from_sphere_coords_to_vec(sphere_coords)
                new_params = np.add(new_params, [0]*num_static_params + perturb_vec)
                new_params_clipped = map(clip_to_unit_line, new_params)

                # if the params + perturb_vec would go out of the unit hypercube then this number will become nonzero
                # and increase with 'severity'. should use this as added, high cost
                missing_param_mass = np.sum(np.power(np.subtract(new_params, new_params_clipped), 2))

                pdf_new.params2matrix_incremental(new_params_clipped)

                pdf_new_perturbs_only = pdf_new[range(len(subject_variables),
                                                      len(subject_variables) + len(perturbed_variables))]
                params_new_perturbs_only = pdf_new_perturbs_only.matrix2params_incremental()

                marginal_pdf_perturbs_diff = np.sum(np.power(np.subtract(params_perturbs_only, params_new_perturbs_only), 2))

                # cost term for going out of the hypercube, which we don't want
                cost_out_of_bounds = np.sqrt(missing_param_mass) / perturbation_size

                if only_non_local or return_cost_list:
                    # note: assumed to be maximally <perturbation_size, namely by a vector on the rim of the hypercube
                    # and then in the perpendicular direction outward into 'unallowed' space (like negative)
                    cost_diff_marginal_perturbed = np.sqrt(marginal_pdf_perturbs_diff) / perturbation_size
                else:
                    # we don't care about a cost term for keeping the marginal the same
                    cost_diff_marginal_perturbed = 0

                if not return_cost_list:
                    return float(rel_weight_out_of_bounds * cost_out_of_bounds + cost_diff_marginal_perturbed)
                else:
                    # this is for diagnostics only, not for any optimization procedure
                    return (cost_out_of_bounds, cost_diff_marginal_perturbed)

            num_sphere_coords = num_free_params - 1  # see description for from_sphere_coords_to_vec()

            initial_perturb_vec = np.random.random(num_sphere_coords)

            optres = minimize(cost_perturb_vec, initial_perturb_vec, bounds=[(0.0, 1.0)]*num_sphere_coords)

            assert optres.success, 'scipy\'s minimize() failed'

            assert optres.fun >= -0.0001, 'cost function as constructed cannot be negative, what happened?'

            # convert optres.x to new parameters
            new_params = params_subs_perturbs  # old param values
            perturb_vec = from_sphere_coords_to_vec(optres.x)
            new_params = np.add(new_params, [0]*num_static_params + perturb_vec)
            new_params_clipped = map(clip_to_unit_line, new_params)

            self.params2matrix_incremental(new_params_clipped)

            # get list of individual cost terms, for better diagnostics by caller
            cost = cost_perturb_vec(optres.x, return_cost_list=True)

            resp = self.PerturbNonLocalResponse()
            resp.pdf = self  # final resulting pdf P'(X,Y), which is slightly different, perturbed version of <self>
            resp.cost_out_of_bounds = float(cost[0])  # cost of how different marginal P(Y) became (bad)
            resp.cost_diff_marginal_perturbed = float(cost[1])  # cost of how different P(Y|X) is, compared to desired difference
            resp.perturb_size = np.linalg.norm(np.subtract(params_subs_perturbs, new_params_clipped))

            return resp
        else:
            pdf_reordered = self.copy()

            retained_vars = list(subject_variables) + list(perturbed_variables)
            ignored_vars = list(np.setdiff1d(range(len(self)), retained_vars))

            if len(ignored_vars) > 0:
                cond_pdf_ignored = self.conditional_probability_distributions(ignored_vars)

            pdf_reordered.reorder_variables(list(subject_variables) + list(perturbed_variables))

            resp = pdf_reordered.perturb(range(len(subject_variables),
                                               len(subject_variables) + len(perturbed_variables)))

            # note: pdf_reordered is now changed in place, the perturbed values

            if len(ignored_vars) > 0:
                pdf_reordered.append_variables_using_conditional_distributions(cond_pdf_ignored)

            reverse_ordering = [-1]*len(self)

            for new_six in xrange(len(subject_variables)):
                orig_six = subject_variables[new_six]
                reverse_ordering[orig_six] = new_six

            for new_pix in xrange(len(subject_variables), len(subject_variables) + len(perturbed_variables)):
                orig_pix = perturbed_variables[new_pix - len(subject_variables)]
                reverse_ordering[orig_pix] = new_pix

            for new_iix in xrange(len(subject_variables) + len(perturbed_variables),
                                  len(subject_variables) + len(perturbed_variables) + len(ignored_vars)):
                orig_pix = perturbed_variables[new_pix - len(subject_variables) - len(perturbed_variables)]
                reverse_ordering[orig_pix] = new_pix

            pdf_reordered.reorder_variables(reverse_ordering)

            self.duplicate(pdf_reordered)  # change in-place this object now (could remove the use of pdf_reordered?)

            return resp


        # assert False, 'todo'
        #
        # num_input_variables = len(self) - num_output_variables
        #
        # assert num_input_variables > 0, 'makes no sense to perturb a relation with an empty set'
        #
        # original_params = self.matrix2params_incremental()
        #
        # static_params = list(self[range(num_input_variables)].matrix2params_incremental())
        #
        # num_free_params = len(original_params) - len(static_params)
        #
        # marginal_output_pdf = self[range(num_input_variables, len(self))]
        # assert len(marginal_output_pdf) == num_output_variables, 'programming error'
        # marginal_output_pdf_params = marginal_output_pdf.matrix2params_incremental()
        #
        # pdf_new = self.copy()  # just to create an object which I can replace everytime in cost function
        #
        # def clip_to_unit_line(num):  # helper function, make sure all probabilities remain valid
        #     return max(min(num, 1), 0)
        #
        # def cost_perturb_non_local(free_params, return_cost_list=False):
        #     new_params = static_params + list(map(clip_to_unit_line, free_params))
        #
        #     pdf_new.params2matrix_incremental(new_params)
        #
        #     marginal_output_pdf_new = pdf_new[range(num_input_variables, len(self))]
        #     marginal_output_pdf_new_params = marginal_output_pdf_new.matrix2params_incremental()
        #
        #     cost_same_output_marginal = np.linalg.norm(np.subtract(marginal_output_pdf_new_params,
        #                                                            marginal_output_pdf_params))
        #     cost_different_relation = np.linalg.norm(np.subtract(free_params, original_params[len(static_params):]))
        #
        #     if not return_cost_list:
        #         cost = np.power(cost_same_output_marginal - 0.0, 2) \
        #                + np.power(cost_different_relation - perturbation_size, 2)
        #     else:
        #         cost = [np.power(cost_same_output_marginal - 0.0, 2),
        #                 np.power(cost_different_relation - perturbation_size, 2)]
        #
        #     return cost
        #
        # initial_guess_perturb_vec = np.random.random(num_free_params)
        # initial_guess_perturb_vec /= np.linalg.norm(initial_guess_perturb_vec)
        # initial_guess_perturb_vec *= perturbation_size
        #
        # initial_guess = np.add(original_params[len(static_params):],
        #                        initial_guess_perturb_vec)
        # initial_guess = map(clip_to_unit_line, initial_guess)  # make sure stays in hypercube's unit volume
        #
        # optres = minimize(cost_perturb_non_local, initial_guess, bounds=[(0.0, 1.0)]*num_free_params)
        #
        # assert optres.success, 'scipy\'s minimize() failed'
        #
        # assert optres.fun >= -0.0001, 'cost function as constructed cannot be negative, what happened?'
        #
        # pdf_new.params2matrix_incremental(list(static_params) + list(optres.x))
        #
        # # get list of individual cost terms, for better diagnostics by caller
        # cost = cost_perturb_non_local(optres.x, return_cost_list=True)
        #
        # resp = self.PerturbNonLocalResponse()
        # resp.pdf = pdf_new  # final resulting pdf P'(X,Y), which is slightly different, perturbed version of <self>
        # resp.cost_same_output_marginal = float(cost[0])  # cost of how different marginal P(Y) became (bad)
        # resp.cost_different_relation = float(cost[1])  # cost of how different P(Y|X) is, compared to desired difference
        #
        # return resp


    def perturb_non_local(self, num_output_variables, perturbation_size=0.001):  # todo: generalize
        """
        Perturb the pdf P(X,Y) by changing P(Y|X) without actually changing P(X) or P(Y), by numerical optimization.
        Y is assumed to be formed by stochastic variables collected at the end of the pdf.
        :param num_output_variables: |Y|.
        :param perturbation_size: the higher, the more different P(Y|X) will be from <self>
        :rtype: PerturbNonLocalResponse
        """
        num_input_variables = len(self) - num_output_variables

        assert num_input_variables > 0, 'makes no sense to perturb a relation with an empty set'

        original_params = self.matrix2params_incremental()

        static_params = list(self[range(num_input_variables)].matrix2params_incremental())

        num_free_params = len(original_params) - len(static_params)

        marginal_output_pdf = self[range(num_input_variables, len(self))]
        assert len(marginal_output_pdf) == num_output_variables, 'programming error'
        marginal_output_pdf_params = marginal_output_pdf.matrix2params_incremental()

        pdf_new = self.copy()  # just to create an object which I can replace everytime in cost function

        def clip_to_unit_line(num):  # helper function, make sure all probabilities remain valid
            return max(min(num, 1), 0)

        def cost_perturb_non_local(free_params, return_cost_list=False):
            new_params = static_params + list(map(clip_to_unit_line, free_params))

            pdf_new.params2matrix_incremental(new_params)

            marginal_output_pdf_new = pdf_new[range(num_input_variables, len(self))]
            marginal_output_pdf_new_params = marginal_output_pdf_new.matrix2params_incremental()

            cost_same_output_marginal = np.linalg.norm(np.subtract(marginal_output_pdf_new_params,
                                                                   marginal_output_pdf_params))
            cost_different_relation = np.linalg.norm(np.subtract(free_params, original_params[len(static_params):]))

            if not return_cost_list:
                cost = np.power(cost_same_output_marginal - 0.0, 2) \
                       + np.power(cost_different_relation - perturbation_size, 2)
            else:
                cost = [np.power(cost_same_output_marginal - 0.0, 2),
                        np.power(cost_different_relation - perturbation_size, 2)]

            return cost

        initial_guess_perturb_vec = np.random.random(num_free_params)
        initial_guess_perturb_vec /= np.linalg.norm(initial_guess_perturb_vec)
        initial_guess_perturb_vec *= perturbation_size

        initial_guess = np.add(original_params[len(static_params):],
                               initial_guess_perturb_vec)
        initial_guess = map(clip_to_unit_line, initial_guess)  # make sure stays in hypercube's unit volume

        optres = minimize(cost_perturb_non_local, initial_guess, bounds=[(0.0, 1.0)]*num_free_params)

        assert optres.success, 'scipy\'s minimize() failed'

        assert optres.fun >= -0.0001, 'cost function as constructed cannot be negative, what happened?'

        pdf_new.params2matrix_incremental(list(static_params) + list(optres.x))

        # get list of individual cost terms, for better diagnostics by caller
        cost = cost_perturb_non_local(optres.x, return_cost_list=True)

        resp = self.PerturbNonLocalResponse()
        resp.pdf = pdf_new  # final resulting pdf P'(X,Y), which is slightly different, perturbed version of <self>
        resp.cost_same_output_marginal = float(cost[0])  # cost of how different marginal P(Y) became (bad)
        resp.cost_different_relation = float(cost[1])  # cost of how different P(Y|X) is, compared to desired difference

        return resp


    def append_globally_resilient_variables(self, num_appended_variables, target_mi):

        # input_variables = [d for d in xrange(self.numvariables) if not d in output_variables]

        parameter_values_before = list(self.matrix2params_incremental())

        pdf_new = self.copy()
        pdf_new.append_variables(num_appended_variables)

        assert pdf_new.numvariables == self.numvariables + num_appended_variables

        parameter_values_after = pdf_new.matrix2params_incremental()

        # this many parameters (each in [0,1]) must be optimized
        num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

        def cost_func_resilience_and_mi(free_params, parameter_values_before):
            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))

            mi = pdf_new.mutual_information(range(len(self)), range(len(self), len(pdf_new)))

            susceptibility = pdf_new.susceptibility_global(num_appended_variables)

            return np.power(abs(target_mi - mi) / target_mi + susceptibility, 2)

        self.append_optimized_variables(num_appended_variables, cost_func=cost_func_resilience_and_mi,
                                        initial_guess=np.random.random(num_free_parameters))

        return


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


    def append_variables_with_target_mi_and_marginal(self, num_appended_variables, target_mi, marginal_probs,
                                                     relevant_variables='all', verbose=False, num_repeats=None):

        # input_variables = [d for d in xrange(self.numvariables) if not d in output_variables]

        if relevant_variables in ('all', 'auto'):
            relevant_variables = range(len(self))
        else:
            assert len(relevant_variables) < len(self), 'cannot be relative to more variables than I originally had'
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

        def cost_func_target_mi2(free_params, parameter_values_before):

            assert np.all(np.isfinite(free_params)), 'looking for bug 23142'
            # assert np.all(np.isfinite(parameter_values_before))  # todo: remove

            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))

            mi = pdf_new.mutual_information(relevant_variables, range(len(self), len(pdf_new)))
            pdf_B = pdf_new[range(len(pdf_new)-num_appended_variables, len(pdf_new))]
            diff_prob_cost = np.mean(np.power(pdf_B.joint_probabilities.joint_probabilities
                                              - marginal_probs, 2))

            return np.power((target_mi - mi) / target_mi, 2) + diff_prob_cost

        self.append_optimized_variables(num_appended_variables, cost_func=cost_func_target_mi2,
                                        initial_guess=np.random.random(num_free_parameters),
                                        verbose=verbose, num_repeats=num_repeats)

        return  # nothing, in-place


    def append_unique_individual_variable(self, about_variable_ix, verbose=True, tol_nonunique=0.05,
                                          num_repeats=3, agnostic_about=None, ignore_variables=None):
        """

        :param about_variable_ix:
        :param verbose:
        :param tol_nonunique:
        :param num_repeats: seems necessary to repeat so maybe 3 or 5?
        :param agnostic_about: I thought this was needed but now not sure (see note inside unique_individual_information())
        :param ignore_variables: used by unique_individual_information() for I_{unq}(X --> Y) to find URVs for only X
        :return:
        """
        assert not np.isscalar(agnostic_about), 'agnostic_about should be a list of ints like [3,4]'

        exponent = 1  # for individual cost terms

        pdf_new = self.copy()
        pdf_new.append_variables(1) # just to get the correct size

        if ignore_variables is None:
            ignore_variables = []

        assert not about_variable_ix in ignore_variables, 'makes no sense'

        def cost_func_unique(free_params, parameter_values_before):
            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))

            mi_indiv = pdf_new.mutual_information([about_variable_ix], [len(self)])  # this is a good thing
            # this is not a good thing:
            mi_nonunique = pdf_new.mutual_information([i for i in xrange(len(self))
                                                       if i != about_variable_ix and not i in ignore_variables],
                                                      [len(self)])

            if agnostic_about is None:
                # I treat the two terms equally important even though mi_nonunique can typically be much higher,
                # so no normalization for the number of 'other' variables.
                # note: I want to have max `tol_nonunique` fraction of nonunique MI so try to get this by weighting:
                cost = -np.power(mi_indiv, exponent) * tol_nonunique + np.power(mi_nonunique, exponent) * (1. - tol_nonunique)
            else:
                mi_agn = pdf_new.mutual_information([len(self)], agnostic_about)

                cost = -np.power(mi_indiv, exponent) * tol_nonunique + 0.5 * (np.power(mi_nonunique, exponent) + np.power(mi_agn, exponent)) * (1. - tol_nonunique)

            # note: if exponent==1 then cost should be <0 to be 'acceptable'
            return cost

        pdf_c = self.copy()
        optres = pdf_c.append_optimized_variables(1, cost_func_unique, verbose=verbose, num_repeats=num_repeats)

        if optres.success:
            self.duplicate(pdf_c)
            return optres
        else:
            raise UserWarning('optimization was unsuccessful: ' + str(optres))


    def append_optimized_variables(self, num_appended_variables, cost_func, initial_guess=None, verbose=True,
                                   num_repeats=None):
        """
        Append variables in such a way that their conditional pdf with the existing variables is optimized in some
        sense, for instance they can be synergistic (append_synergistic_variables) or orthogonalized
        (append_orthogonalized_variables). Use the cost_func to determine the relation between the new appended
        variables and the pre-existing ones.
        :param num_appended_variables:
        :param cost_func: a function cost_func(free_params, parameter_values_before) which returns a float.
        The parameter set list(parameter_values_before) + list(free_params) defines a joint pdf of the appended
        variables together with the pre-existing ones, and free_params by itself defines completely the conditional
        pdf of the new variables given the previous. Use params2matrix_incremental to construct a joint pdf from the
        parameters and evaluate whatever you need, and then return a float. The higher the return value of cost_func
        the more desirable the joint pdf induced by the parameter set list(parameter_values_before) + list(free_params).
        :param initial_guess: initial guess for 'free_params' where you think cost_func will return a relatively
        low value. It can also be None, in which case a random point in parameter space will be chosen. It can also
        be an integer value like 10, in which case 10 optimizations will be run each starting from a random point
        in parameter space, and the best solution is selected.
        :param verbose:
        :rtype: scipy.optimize.OptimizeResult
        """

        # these parameters should be unchanged and the first set of parameters of the resulting pdf_new
        parameter_values_before = list(self.matrix2params_incremental())

        assert min(parameter_values_before) >= -0.00000001, \
            'minimum of %s is < 0, should not be.' % parameter_values_before
        assert max(parameter_values_before) <= 1.00000001, \
            'minimum of %s is < 0, should not be.' % parameter_values_before

        if __debug__:
            debug_params_before = copy.deepcopy(parameter_values_before)

        # a pdf with XORs as appended variables (often already MSRV for binary variables), good initial guess?
        # note: does not really matter how I set the pdf of this new pdf, as long as it has the correct number of
        # paarameters for optimization below
        pdf_new = self.copy()
        pdf_new.append_variables_using_state_transitions_table(
            state_transitions=lambda vals, mv: [int(np.mod(np.sum(vals), mv))]*num_appended_variables)

        assert pdf_new.numvariables == self.numvariables + num_appended_variables

        parameter_values_after = pdf_new.matrix2params_incremental()

        assert num_appended_variables > 0, 'makes no sense to add 0 variables'
        assert len(parameter_values_after) > len(parameter_values_before), 'should be >0 free parameters to optimize?'
        # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
        # have to optimize the latter part of parameter_values_after
        np.testing.assert_array_almost_equal(parameter_values_before,
                                             parameter_values_after[:len(parameter_values_before)])

        # this many parameters (each in [0,1]) must be optimized
        num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

        assert num_appended_variables == 0 or num_free_parameters > 0

        # if initial_guess is None:
        #     initial_guess = np.random.random(num_free_parameters)  # start from random point in parameter space

        param_vectors_trace = []  # storing the parameter vectors visited by the minimize() function

        if num_repeats is None:
            if type(initial_guess) == int:
                num_repeats = int(initial_guess)
                initial_guess = None

                assert num_repeats > 0, 'makes no sense to optimize zero times?'
            else:
                num_repeats = 1

        optres = None

        def cost_func_wrapper(free_params, parameter_values_before):
            # note: jezus CHRIST not only does minimize() ignore the bounds I give it, it also suggests [nan, ...]!
            # assert np.all(np.isfinite(free_params)), 'looking for bug 23142'
            if not np.all(np.isfinite(free_params)):
                return np.power(np.sum(np.isfinite(free_params)), 2) * 10.
            else:
                clipped_free_params = np.max([np.min([free_params, np.ones(np.shape(free_params))], axis=0),
                                             np.zeros(np.shape(free_params))], axis=0)
                # penalize going out of bounds
                extra_cost = np.power(np.sum(np.abs(np.subtract(free_params, clipped_free_params))), 2)
                return cost_func(clipped_free_params, parameter_values_before) + extra_cost

        for rep in xrange(num_repeats):
            if initial_guess is None:
                initial_guess_i = np.random.random(num_free_parameters)  # start from random point in parameter space
            else:
                initial_guess_i = initial_guess  # always start from supplied point in parameter space

            assert len(initial_guess_i) == num_free_parameters
            assert np.all(np.isfinite(initial_guess_i)), 'looking for bug 55142'
            assert np.all(np.isfinite(parameter_values_before)), 'looking for bug 44142'

            if verbose:
                print 'debug: starting minimize() #' + str(rep) \
                      + ' at params=' + str(initial_guess_i) + ' at cost_func=' \
                      + str(cost_func_wrapper(initial_guess_i, parameter_values_before))

            optres_i = minimize(cost_func_wrapper,
                              initial_guess_i, bounds=[(0.0, 1.0)]*num_free_parameters,
                              # callback=(lambda xv: param_vectors_trace.append(list(xv))) if verbose else None,
                              args=(parameter_values_before,))

            if optres_i.success:
                if verbose:
                    print 'debug: successfully ended minimize() #' + str(rep) \
                          + ' at params=' + str(optres_i.x) + ' at cost_func=' \
                          + str(optres_i.fun)

                if optres is None:
                    optres = optres_i
                elif optres.fun > optres_i.fun:
                    optres = optres_i
                else:
                    pass  # this solution is worse than before, so do not change optres

        if optres is None:
            # could never find a good solution, in all <num_repeats> attempts
            raise UserWarning('always failed to successfully optimize: increase num_repeats')

        assert len(optres.x) == num_free_parameters
        assert max(optres.x) <= 1.0001, 'parameter bound significantly violated: ' + str(optres.x)
        assert min(optres.x) >= -0.0001, 'parameter bound significantly violated: ' + str(optres.x)

        # clip the parameters within the allowed bounds
        optres.x = [min(max(xi, 0.0), 1.0) for xi in optres.x]

        optimal_parameters_joint_pdf = list(parameter_values_before) + list(optres.x)

        assert min(optimal_parameters_joint_pdf) >= 0.0, \
            'minimum of %s is < 0, should not be.' % optimal_parameters_joint_pdf
        assert min(optimal_parameters_joint_pdf) <= 1.0, \
            'minimum of %s is > 1, should not be.' % optimal_parameters_joint_pdf
        assert min(parameter_values_before) >= 0.0, \
            'minimum of %s is < 0, should not be.' % parameter_values_before
        assert min(parameter_values_before) <= 1.0, \
            'minimum of %s is > 1, should not be.' % parameter_values_before

        pdf_new.params2matrix_incremental(optimal_parameters_joint_pdf)

        assert len(pdf_new) == len(self) + num_appended_variables

        if __debug__:
            parameter_values_after2 = pdf_new.matrix2params_incremental()

            assert len(parameter_values_after2) > len(parameter_values_before), 'should be additional free parameters'
            # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
            # have to optimize the latter part of parameter_values_after
            np.testing.assert_array_almost_equal(parameter_values_before,
                                                 parameter_values_after2[:len(parameter_values_before)])
            # note: for the if see the story in params2matrix_incremental()
            if not (0.000001 >= min(self.scalars_up_to_level(parameter_values_after2)) or \
                            0.99999 <= max(self.scalars_up_to_level(parameter_values_after2))):
                try:
                    np.testing.assert_array_almost_equal(parameter_values_after2[len(parameter_values_before):],
                                                         optres.x)
                except AssertionError as e:
                    # are they supposed to be equal, but in different order?
                    print 'debug: sum params after 1 =', np.sum(parameter_values_after2[len(parameter_values_before):])
                    print 'debug: sum params after 2 =', optres.x
                    print 'debug: parameter_values_before (which IS equal and correct) =', parameter_values_before
                    # does this one below have a 1 or 0 in it? because then the error could be caused by the story in
                    # params2matrix_incremental()
                    print 'debug: parameter_values_after2 =', parameter_values_after2

                    raise AssertionError(e)
        if __debug__:
            # unchanged, not accidentally changed by passing it as reference? looking for bug
            np.testing.assert_array_almost_equal(debug_params_before, parameter_values_before)

        self.duplicate(pdf_new)

        return optres


    def append_orthogonalized_variables(self, variables_to_orthogonalize, num_added_variables_orthogonal,
                                        num_added_variables_parallel, verbose=True,
                                        num_repeats=1, randomization_per_repeat=0.01):

        """
        Let X=<variables_to_orthogonalize> and Y=complement[<variables_to_orthogonalize>]. Add two sets of variables
        X1 and X2 such that I(X1:Y)=0, I(X1:X)=H(X|Y); and I(X2:Y)=I(X2:X)=I(X:Y), and I(X1:X2)=0. In words, X is being
        decomposed into two parts: X1 (orthogonal to Y, MI=0) and X2 (parallel to Y, MI=max).

        This object itself will be expanded by <num_added_variables_orthogonal> + <num_added_variables_parallel>
        variables.

        Warning: if this pdf object also contains other variables Z which the orthogonalization can ignore, then
        first take them out of the pdf (marginalize or condition) because it blows up the number of free parameters
        that this function must optimize.

        (Do the optimization jointly, so both parallel and orthogonal together, because the function below seems not so
        effective for some reason.)
        :param variables_to_orthogonalize: list of variable indices which should be decomposed (X). The complemennt in
        range(len(self)) is then implicitly the object to decompose against ((Y in the description).
        :type variables_to_orthogonalize: list of int
        :param num_added_variables_orthogonal: |X1|
        :param num_added_variables_parallel: |X2|
        :param verbose:
        :param num_repeats: number of times to perform minimize() starting from random parameters
        and take the best result.
        :param randomization_per_repeat:
        :return: :raise ValueError:
        """
        assert num_added_variables_orthogonal > 0, 'cannot make parallel variables if I don\'t first make orthogonal'
        assert num_added_variables_parallel > 0, 'this may not be necessary, assert can be removed at some point?'

        # remove potential duplicates; and sort (for no reason)
        variables_to_orthogonalize = sorted(list(set(variables_to_orthogonalize)))

        pdf_ortho_para = self.copy()

        # these parameters should remain unchanged during the optimization, these are the variables (Y, X) where
        # X is the variable to be orthogonalized into the added (X1, X2) and Y is to be orthogonalized against.
        original_parameters = range(len(pdf_ortho_para.matrix2params_incremental(return_flattened=True)))

        original_variables = range(len(self))  # this includes variables_to_orthogonalize
        subject_variables = list(np.setdiff1d(original_variables, variables_to_orthogonalize))

        # the rest of the function assumes that the variables_to_orthogonalize are all at the end of the
        # original_variables, so that appending a partial conditional pdf (conditioned on
        # len(variables_to_orthogonalize) variables) will be conditioned on the variables_to_orthogonalize only. So
        # reorder the variables and recurse to this function once.
        if not max(subject_variables) < min(variables_to_orthogonalize):
            pdf_reordered = self.copy()

            pdf_reordered.reorder_variables(subject_variables + variables_to_orthogonalize)

            assert len(pdf_reordered) == len(self)
            assert len(range(len(subject_variables), pdf_reordered.numvariables)) == len(variables_to_orthogonalize)

            # perform orthogonalization on the ordered pdf, which it is implemented for
            pdf_reordered.append_orthogonalized_variables(range(len(subject_variables), len(self)))

            # find the mapping from ordered to disordered, so that I can reverse the reordering above
            new_order = subject_variables + variables_to_orthogonalize
            original_order = [-1] * len(new_order)
            for varix in xrange(len(new_order)):
                original_order[new_order[varix]] = varix

            assert not -1 in original_order
            assert max(original_order) < len(original_order), 'there were only so many original variables...'

            pdf_reordered.reorder_variables(original_order + range(len(original_order), len(pdf_reordered)))

            self.duplicate(pdf_reordered)

            # due to the scipy's minimize() procedure the list of parameters can be temporarily slightly out of bounds, like
            # 1.000000001, but soon after this should be clipped to the allowed range and e.g. at this point the parameters
            # are expected to be all valid
            assert min(self.matrix2params_incremental()) >= 0.0, \
                'parameter(s) out of bound: ' + str(self.matrix2params_incremental())
            assert max(self.matrix2params_incremental()) <= 1.0, \
                'parameter(s) out of bound: ' + str(self.matrix2params_incremental())

            return

        assert len(np.intersect1d(original_variables, variables_to_orthogonalize)) == len(variables_to_orthogonalize), \
            'original_variables should include the variables_to_orthogonalize'

        pdf_ortho_para.append_variables(num_added_variables_orthogonal)
        orthogonal_variables = range(len(self), len(pdf_ortho_para))

        orthogonal_parameters = range(len(original_parameters),
                                      len(pdf_ortho_para.matrix2params_incremental(return_flattened=True)))

        pdf_ortho_para.append_variables(num_added_variables_parallel)
        parallel_variables = range(len(pdf_ortho_para) - num_added_variables_parallel, len(pdf_ortho_para))

        parallel_parameters = range(len(orthogonal_parameters) + len(original_parameters),
                                    len(pdf_ortho_para.matrix2params_incremental(return_flattened=True)))

        assert len(np.intersect1d(orthogonal_parameters, parallel_parameters)) == 0, \
            'error: orthogonal_parameters = ' + str(orthogonal_parameters) + ', parallel_parameters = ' \
            + str(parallel_parameters)
        assert len(np.intersect1d(orthogonal_variables, parallel_variables)) == 0

        free_parameters = list(orthogonal_parameters) + list(parallel_parameters)

        initial_params_list = np.array(list(pdf_ortho_para.matrix2params_incremental(return_flattened=True)))

        # todo: make this more efficient by having only free parameters X1,X2 as p(X1,X2|X), not also conditioned on Y?
        # this would make it a two-step thing maybe again naively, but maybe there is a way to still do it simultaneous
        # (of course the cost function DOES depend on Y, but X1,X2=f(X) only and the point is to find an optimal f(),)

        # todo: START of optimization

        # let the X1,X2 parameterization only depend on X (=variables_to_orthogonalize) to reduce the parameter space
        # greatly
        pdf_X = self.marginalize_distribution(variables_to_orthogonalize)

        pdf_X_X1_X2 = pdf_X.copy()
        pdf_X_X1_X2.append_variables(num_added_variables_orthogonal)
        pdf_X_X1_X2.append_variables(num_added_variables_parallel)

        free_params_X1_X2_given_X = range(len(pdf_X.matrix2params_incremental()),
                                          len(pdf_X_X1_X2.matrix2params_incremental()))
        # parameter values at these parameter indices should not change:
        static_params_X = range(len(pdf_X.matrix2params_incremental()))
        static_params_X_values = list(np.array(pdf_X.matrix2params_incremental())[static_params_X])

        # check if building a complete joint pdf using the subset-conditional pdf works as expected
        if __debug__:
            cond_pdf_X1_X2_given_X = pdf_X_X1_X2.conditional_probability_distributions(range(len(variables_to_orthogonalize)))

            assert cond_pdf_X1_X2_given_X.num_output_variables() == num_added_variables_orthogonal \
                                                              + num_added_variables_parallel

            pdf_test_Y_X_X1_X2 = self.copy()
            pdf_test_Y_X_X1_X2.append_variables_using_conditional_distributions(cond_pdf_X1_X2_given_X)

            # test if first so-many params are the same as pdf_ortho_para's
            np.testing.assert_array_almost_equal(initial_params_list[:len(original_parameters)],
                                                 pdf_test_Y_X_X1_X2.matrix2params_incremental()[:len(original_parameters)])
            # still, it would be immensely unlikely that the two pdfs are the same, since the conditional pdf(X1,X2|X)
            # is independently and randomly generated for both pdfs
            assert pdf_test_Y_X_X1_X2 != pdf_ortho_para

        # used repeatedly in the cost function below, prevent recomputing it every time
        ideal_H_X1 = self.conditional_entropy(variables_to_orthogonalize, subject_variables)
        ideal_mi_X2_Y = self.mutual_information(variables_to_orthogonalize, subject_variables)

        default_weights = (1, 1, 1, 1, 1, 1)  # used for cost_func_minimal_params and to know the number of weights


        # cost function used in scipy's minimize() procedure
        def cost_func_minimal_params(proposed_params, rel_weights=default_weights):
            assert len(proposed_params) == len(free_params_X1_X2_given_X)

            # relative weight coefficients for the different terms that contribute to the cost function below.
            wIso, wIvo, wIvp, wIsp, wHop, wIop = map(abs, rel_weights)

            pdf_X_X1_X2.params2matrix_incremental(static_params_X_values + list(proposed_params))

            cond_pdf_X1_X2_given_X = pdf_X_X1_X2.conditional_probability_distributions(range(len(variables_to_orthogonalize)))
            pdf_proposed_Y_X_X1_X2 = self.copy()
            pdf_proposed_Y_X_X1_X2.append_variables_using_conditional_distributions(cond_pdf_X1_X2_given_X)

            # test if first so-many params are the same as pdf_ortho_para's
            np.testing.assert_array_almost_equal(initial_params_list[:len(original_parameters)],
                                                 pdf_proposed_Y_X_X1_X2.matrix2params_incremental()[:len(original_parameters)])

            # note: unwanted effects should be positive terms ('cost'), and desired MIs should be negative terms
            # note: if you change the cost function here then you should also change the optimal value below
            # cost = wIso * pdf_proposed_Y_X_X1_X2.mutual_information(subject_variables, orthogonal_variables) \
            #        - wIsp * pdf_proposed_Y_X_X1_X2.mutual_information(subject_variables, parallel_variables) \
            #        - wHop * pdf_proposed_Y_X_X1_X2.entropy(orthogonal_variables) \
            #        + wIop * pdf_proposed_Y_X_X1_X2.mutual_information(orthogonal_variables, parallel_variables)

            # - wHop * pdf_proposed_Y_X_X1_X2.entropy(orthogonal_variables + parallel_variables) \

            # note: each term is intended to be normalized to [0, 1], where 0 is best and 1 is worst. Violation of this
            # is possible though, but they are really bad solutions.

            cost_terms = dict()
            cost_terms['Iso'] = \
                wIso * abs(pdf_proposed_Y_X_X1_X2.mutual_information(subject_variables, orthogonal_variables) - 0.0) \
                / ideal_mi_X2_Y
            cost_terms['Ivo'] = \
                wIvo * abs(pdf_proposed_Y_X_X1_X2.mutual_information(variables_to_orthogonalize, orthogonal_variables)
                           - ideal_H_X1) / ideal_H_X1
            # question: is the following one necessary?
            # cost_terms['Ivp'] = \
            #     wIvp * abs(pdf_proposed_Y_X_X1_X2.mutual_information(variables_to_orthogonalize, parallel_variables)
            #                - ideal_mi_X2_Y) / ideal_mi_X2_Y
            cost_terms['Ivp'] = 0.0
            cost_terms['Isp'] = \
                wIsp * abs(pdf_proposed_Y_X_X1_X2.mutual_information(subject_variables, parallel_variables)
                           - ideal_mi_X2_Y) / ideal_mi_X2_Y
            # cost_terms['Hop'] = wHop * abs(pdf_proposed_Y_X_X1_X2.entropy(orthogonal_variables) - ideal_H_X1)
            cost_terms['Hop'] = \
                wHop * 0.0  # let's see what happens now (trying to improve finding global optimum)
            cost_terms['Iop'] = \
                wIop * abs(pdf_proposed_Y_X_X1_X2.mutual_information(orthogonal_variables, parallel_variables) - 0.0) \
                / ideal_H_X1

            # sum of squared errors, or norm of vector in error space, to make a faster convergence hopefully
            cost = float(np.sum(np.power(cost_terms.values(), 2)))

            assert np.isfinite(cost)
            assert np.isscalar(cost)

            return float(cost)


        if __debug__:
            # for each term in the cost function above I determine what would be the optimal value,
            # note: if you change the cost function above then you should also change this optimal value
            debug_optimal_cost_value = 0

            # hack, I should actually perform some min()'s in this but this is just a rough guide:
            # note: assuming orthogonal_variables = variables_to_orthogonalize and H(parallel_variables) = 0
            debug_worst_cost_value = ideal_mi_X2_Y + abs(self.entropy(variables_to_orthogonalize) - ideal_H_X1) \
                                     + ideal_mi_X2_Y + ideal_mi_X2_Y \
                                     + abs(self.entropy(variables_to_orthogonalize) - ideal_H_X1)
                                                   # + ideal_mi_X2_Y \
                               # + 2 * abs(self.entropy(variables_to_orthogonalize) - ideal_H_X1) + ideal_mi_X2_Y

            debug_random_time_before = time.time()

            debug_num_random_costs = 20
            debug_random_cost_values = [cost_func_minimal_params(np.random.random(len(free_params_X1_X2_given_X)))
                                  for _ in xrange(debug_num_random_costs)]

            debug_random_time_after = time.time()

            # for trying to eye-ball whether the minimize() is doing significantly better than just random sampling,
            # getting a feel for the solution space
            debug_avg_random_cost_val = np.mean(debug_random_cost_values)
            debug_std_random_cost_val = np.std(debug_random_cost_values)
            debug_min_random_cost_val = np.min(debug_random_cost_values)
            debug_max_random_cost_val = np.max(debug_random_cost_values)

            if verbose:
                print 'debug: cost values of random parameter vectors:', debug_random_cost_values, '-- took', \
                    debug_random_time_after - debug_random_time_before, 'seconds for', debug_num_random_costs, \
                    'vectors.'

        initial_guess_minimal = list(np.array(pdf_X_X1_X2.matrix2params_incremental())[free_params_X1_X2_given_X])

        # todo: bring optimal_cost_value to release code (not __debug__) and use it as criterion to decide if the
        # solution is acceptable? Maybe pass a tolerance value in which case I would raise an exception if exceeded?

        if verbose and __debug__:
            print 'debug: append_orthogonalized_variables: BEFORE optimization, cost func =', \
                cost_func_minimal_params(initial_guess_minimal), '(optimal=' + str(debug_optimal_cost_value) \
                + ', a worst=' + str(debug_worst_cost_value) + ', avg~=' + str(debug_avg_random_cost_val) \
                                                                 + '+-' + str(debug_std_random_cost_val) + ')'

            # mutual informations
            print 'debug: pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables) =', \
                pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables)
            print 'debug: pdf_ortho_para.mutual_information(subject_variables, parallel_variables) =', \
                pdf_ortho_para.mutual_information(subject_variables, parallel_variables)
            print 'debug: pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables) =', \
                pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables)
            print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables)
            print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, parallel_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, parallel_variables)
            print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
                  'orthogonal_variables + parallel_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables + parallel_variables)
            print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
                  'subject_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, subject_variables)

            # entropies
            print 'debug: pdf_ortho_para.entropy(subject_variables) =', \
                pdf_ortho_para.entropy(subject_variables)
            print 'debug: pdf_ortho_para.entropy(variables_to_orthogonalize) =', \
                pdf_ortho_para.entropy(variables_to_orthogonalize)
            print 'debug: pdf_ortho_para.entropy(orthogonal_variables) =', \
                pdf_ortho_para.entropy(orthogonal_variables)
            print 'debug: pdf_ortho_para.entropy(parallel_variables) =', \
                pdf_ortho_para.entropy(parallel_variables)

            print 'debug: (num free parameters for optimization:', len(free_params_X1_X2_given_X), ')'

            time_before = time.time()

        if num_repeats == 1:
            optres = minimize(cost_func_minimal_params, initial_guess_minimal,
                              bounds=[(0.0, 1.0)]*len(free_params_X1_X2_given_X),
                              args=((1,)*len(default_weights),))
        elif num_repeats > 1:
            # perform the minimization <num_repeats> times starting from random points in parameter space and select
            # the best one (lowest cost function value)

            # note: the args= argument passes the relative weights of the different (now 4) terms in the cost function
            # defined above. At every next iteration it allows more and more randomization around the value (1,1,1,1)
            # which means that every term would be equally important.
            optres_list = [minimize(cost_func_minimal_params, np.random.random(len(initial_guess_minimal)),
                                    bounds=[(0.0, 1.0)]*len(free_params_X1_X2_given_X),
                                    args=(tuple(1.0 + np.random.randn(len(default_weights))
                                                * randomization_per_repeat * repi),))
                           for repi in xrange(num_repeats)]

            if verbose and __debug__:
                print 'debug: num_repeats=' + str(num_repeats) + ', all cost values were: ' \
                      + str([resi.fun for resi in optres_list])
                print 'debug: successes =', [resi.success for resi in optres_list]

            optres_list = [resi for resi in optres_list if resi.success]  # filter out the unsuccessful optimizations

            assert len(optres_list) > 0, 'all ' + str(num_repeats) + ' optimizations using minimize() failed...?!'

            costvals = [res.fun for res in optres_list]
            min_cost = min(costvals)
            optres_ix = costvals.index(min_cost)

            assert optres_ix >= 0 and optres_ix < len(optres_list)

            optres = optres_list[optres_ix]
        else:
            raise ValueError('cannot repeat negative times')

        assert optres.success, 'scipy\'s minimize() failed'

        assert optres.fun >= -0.0001, 'cost function as constructed cannot be negative, what happened?'

        # build the most optimal PDF then finally:
        pdf_X_X1_X2.params2matrix_incremental(static_params_X_values + list(optres.x))
        cond_pdf_X1_X2_given_X = pdf_X_X1_X2.conditional_probability_distributions(range(len(variables_to_orthogonalize)))
        pdf_ortho_para = self.copy()
        pdf_ortho_para.append_variables_using_conditional_distributions(cond_pdf_X1_X2_given_X)

        # test if first so-many params are the same as pdf_ortho_para's
        np.testing.assert_array_almost_equal(initial_params_list[:len(original_parameters)],
                                             pdf_test_Y_X_X1_X2.matrix2params_incremental()[:len(original_parameters)])

        if verbose and __debug__:
            print 'debug: append_orthogonalized_variables: AFTER optimization, cost func =', optres.fun

            # mutual informations
            print 'debug: pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables) =', \
                pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables), \
                '(optimal=' + str(debug_optimal_cost_value) + ')'
            print 'debug: pdf_ortho_para.mutual_information(subject_variables, parallel_variables) =', \
                pdf_ortho_para.mutual_information(subject_variables, parallel_variables)
            print 'debug: pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables) =', \
                pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables)
            print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables)
            print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, parallel_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, parallel_variables)
            print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
                  'orthogonal_variables + parallel_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables + parallel_variables)
            print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
                  'subject_variables) =', \
                pdf_ortho_para.mutual_information(variables_to_orthogonalize, subject_variables)

            # entropies
            print 'debug: pdf_ortho_para.entropy(subject_variables) =', \
                pdf_ortho_para.entropy(subject_variables)
            print 'debug: pdf_ortho_para.entropy(variables_to_orthogonalize) =', \
                pdf_ortho_para.entropy(variables_to_orthogonalize)
            print 'debug: pdf_ortho_para.entropy(orthogonal_variables) =', \
                pdf_ortho_para.entropy(orthogonal_variables)
            print 'debug: pdf_ortho_para.entropy(parallel_variables) =', \
                pdf_ortho_para.entropy(parallel_variables)

            time_after = time.time()

            print 'debug: the optimization took', time_after - time_before, 'seconds in total.'

        self.duplicate(pdf_ortho_para)

        return

        # todo: END of optimization

        # # note: the code below is 'old', in the sense that it should be equivalent above but using an optimization
        # # step in a (much) larger parameter space
        #
        # def cost_func2(proposed_params):
        #     assert len(free_parameters) == len(proposed_params)
        #
        #     params_list = list(initial_params_list[original_parameters]) + list(proposed_params)
        #
        #     pdf_ortho_para.params2matrix_incremental(params_list)  # in-place
        #
        #     # note: unwanted effects should be positive terms ('cost'), and desired MIs should be negative terms
        #     cost = pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables) \
        #            - pdf_ortho_para.mutual_information(subject_variables, parallel_variables) \
        #            - pdf_ortho_para.mutual_information(variables_to_orthogonalize,
        #                                                orthogonal_variables + parallel_variables)
        #
        #     assert np.isfinite(cost)
        #     assert np.isscalar(cost)
        #
        #     return float(cost)
        #
        #
        # initial_guess = list(initial_params_list[free_parameters])
        #
        # if verbose and __debug__:
        #     print 'debug: append_orthogonalized_variables: BEFORE optimization, cost func =', cost_func2(initial_guess)
        #
        #     # mutual informations
        #     print 'debug: pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables) =', \
        #         pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(subject_variables, parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(subject_variables, parallel_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
        #           'orthogonal_variables + parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables + parallel_variables)
        #
        #     # entropies
        #     print 'debug: pdf_ortho_para.entropy(subject_variables) =', \
        #         pdf_ortho_para.entropy(subject_variables)
        #     print 'debug: pdf_ortho_para.entropy(variables_to_orthogonalize) =', \
        #         pdf_ortho_para.entropy(variables_to_orthogonalize)
        #     print 'debug: pdf_ortho_para.entropy(orthogonal_variables) =', \
        #         pdf_ortho_para.entropy(orthogonal_variables)
        #     print 'debug: pdf_ortho_para.entropy(parallel_variables) =', \
        #         pdf_ortho_para.entropy(parallel_variables)
        #
        # optres = minimize(cost_func2, initial_guess, bounds=[(0.0, 1.0)]*len(free_parameters))
        #
        # assert optres.success, 'scipy\'s minimize() failed'
        #
        # if verbose and __debug__:
        #     print 'debug: append_orthogonalized_variables: AFTER optimization, cost func =', optres.fun
        #
        #     # mutual informations
        #     print 'debug: pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables) =', \
        #         pdf_ortho_para.mutual_information(subject_variables, orthogonal_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(subject_variables, parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(subject_variables, parallel_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(orthogonal_variables, parallel_variables)
        #     print 'debug: pdf_ortho_para.mutual_information(variables_to_orthogonalize, ' \
        #           'orthogonal_variables + parallel_variables) =', \
        #         pdf_ortho_para.mutual_information(variables_to_orthogonalize, orthogonal_variables + parallel_variables)
        #
        #     # entropies
        #     print 'debug: pdf_ortho_para.entropy(subject_variables) =', \
        #         pdf_ortho_para.entropy(subject_variables)
        #     print 'debug: pdf_ortho_para.entropy(variables_to_orthogonalize) =', \
        #         pdf_ortho_para.entropy(variables_to_orthogonalize)
        #     print 'debug: pdf_ortho_para.entropy(orthogonal_variables) =', \
        #         pdf_ortho_para.entropy(orthogonal_variables)
        #     print 'debug: pdf_ortho_para.entropy(parallel_variables) =', \
        #         pdf_ortho_para.entropy(parallel_variables)
        #
        # self.duplicate(pdf_ortho_para)
        #
        # # del pdf_ortho_para


    # todo: set default entropy_cost_factor=0.1 or 0.05?
    # def append_orthogonalized_variables(self, variables_to_orthogonalize, num_added_variables_orthogonal=None,
    #                                     num_added_variables_parallel=None, entropy_cost_factor=0.1, verbose=True,
    #                                     num_repeats=3):
    #     """
    #     Orthogonalize the given set of variables_to_orthogonalize=X relative to the rest. I.e.,
    #     decompose X into two parts {X1,X2}
    #     :param num_added_variables_parallel:
    #     :param verbose:
    #     :param num_repeats: number of times that the optimization procedure of the cost function (for both
    #     the orthogonal variables and the parallel variables) is repeated, of which the best solution is then chosen.
    #     where I(X1:rest)=0 but I(X1:X)=H(X1) and I(X2:rest)=H(X2). The orthogonal set is added first and the parallel
    #     set last.
    #
    #     In theory the entropy of self as a whole should not increase, though it is not the end of the world if it does.
    #     But you can set e.g. entropy_cost_factor=0.1 to try and make it increase as little as possible.
    #     :param variables_to_orthogonalize: set of variables to orthogonalize (X)
    #     :param num_added_variables_orthogonal: number of variables in the orthogonal variable set to add. The more the
    #     better, at the combinatorial cost of memory and computation of course, though at some value the benefit should
    #     diminish to zero (or perhaps negative if the optimization procedure sucks at higher dimensions). If you also
    #     add parallel variables (num_added_variables_parallel > 0) then the quality of the parallel variable set depends
    #     on the quality of the orthogonal variable set, so then a too low number for num_added_variables_orthogonal
    #     might hurt.
    #     :type num_added_variables_orthogonal: int
    #     :type num_added_variables_parallel: int
    #     :param entropy_cost_factor: keep entropy_cost_factor < 1 or (close to) 0.0 (not negative)
    #     :type variables: list of int
    #     """
    #
    #     # remove potential duplicates; and sort (for no reason)
    #     variables_to_orthogonalize = sorted(list(set(variables_to_orthogonalize)))
    #
    #     # due to the scipy's minimize() procedure the list of parameters can be temporarily slightly out of bounds, like
    #     # 1.000000001, but soon after this should be clipped to the allowed range and e.g. at this point the parameters
    #     # are expected to be all valid
    #     assert min(self.matrix2params_incremental()) >= 0.0, \
    #         'parameter(s) out of bound: ' + str(self.matrix2params_incremental())
    #     assert max(self.matrix2params_incremental()) <= 1.0, \
    #         'parameter(s) out of bound: ' + str(self.matrix2params_incremental())
    #
    #     # will add two sets of variables, one orthogonal to the pre-existing 'rest' (i.e. MI zero) and one parallel
    #     # (i.e. MI max). How many variables should each set contain?
    #     # note: setting to len(variables_to_orthogonalize) means that each set has at least surely enough entropy to do the job,
    #     # but given the discrete nature I am not sure if it is also always optimal. Can increase this to be more sure?
    #     # At the cost of more memory usage and computational requirements of course.
    #     if num_added_variables_orthogonal is None:
    #         num_added_variables_orthogonal = len(variables_to_orthogonalize)
    #     if num_added_variables_parallel is None:
    #         num_added_variables_parallel = len(variables_to_orthogonalize)
    #
    #     # variables to orthogonalize against
    #     remaining_variables = sorted([varix for varix in xrange(self.numvariables)
    #                                   if not varix in variables_to_orthogonalize])
    #
    #     if not max(remaining_variables) < min(variables_to_orthogonalize):
    #         pdf_reordered = self.copy()
    #
    #         pdf_reordered.reorder_variables(remaining_variables + variables_to_orthogonalize)
    #
    #         assert len(pdf_reordered) == len(self)
    #         assert len(range(len(remaining_variables), pdf_reordered.numvariables)) == len(variables_to_orthogonalize)
    #
    #         # perform orthogonalization on the ordered pdf, which it is implemented for
    #         pdf_reordered.append_orthogonalized_variables(range(len(remaining_variables),
    #                                                             pdf_reordered.numvariables))
    #
    #         # find the mapping from ordered to disordered, so that I can reverse the reordering above
    #         new_order = remaining_variables + variables_to_orthogonalize
    #         original_order = [-1] * len(new_order)
    #         for varix in xrange(len(new_order)):
    #             original_order[new_order[varix]] = varix
    #
    #         assert not -1 in original_order
    #         assert max(original_order) < len(original_order), 'there were only so many original variables...'
    #
    #         pdf_reordered.reorder_variables(original_order + range(len(original_order), len(pdf_reordered)))
    #
    #         self.duplicate(pdf_reordered)
    #
    #         # due to the scipy's minimize() procedure the list of parameters can be temporarily slightly out of bounds, like
    #         # 1.000000001, but soon after this should be clipped to the allowed range and e.g. at this point the parameters
    #         # are expected to be all valid
    #         assert min(self.matrix2params_incremental()) >= 0.0, \
    #             'parameter(s) out of bound: ' + str(self.matrix2params_incremental())
    #         assert max(self.matrix2params_incremental()) <= 1.0, \
    #             'parameter(s) out of bound: ' + str(self.matrix2params_incremental())
    #     else:
    #         pdf_result = self.copy()  # used to store the result and eventually I will copy to self
    #
    #         ### first add the ORTHOGONAL part
    #
    #         # did not yet do minimize() or something like that so why would a parameter already be out of bound
    #         # even if by a small amount)? should be detected already at some earlier stage.
    #         assert min(pdf_result.matrix2params_incremental()) >= 0.0, 'parameter(s) out of bound, weird: ' \
    #                                                                    + str(pdf_result.matrix2params_incremental())
    #         assert max(pdf_result.matrix2params_incremental()) <= 1.0, 'parameter(s) out of bound, weird' \
    #                                                                    + str(pdf_result.matrix2params_incremental())
    #
    #         if num_added_variables_orthogonal > 0:
    #             pdf_for_ortho_only_optimization = self.copy()
    #             pdf_for_ortho_only_optimization.append_variables(num_added_variables_orthogonal)
    #
    #             num_free_parameters_ortho_only = len(pdf_for_ortho_only_optimization.matrix2params_incremental()) \
    #                                                     - len(self.matrix2params_incremental())
    #
    #             assert num_free_parameters_ortho_only > 0 or num_added_variables_orthogonal == 0
    #
    #             orthogonal_variables = range(len(self), len(self) + num_added_variables_orthogonal)
    #
    #             assert len(orthogonal_variables) == num_added_variables_orthogonal
    #
    #             # did not yet do minimize() or something like that so why would a parameter already be out of bound
    #             # even if by a small amount)? should be detected already at some earlier stage.
    #             assert min(pdf_for_ortho_only_optimization.matrix2params_incremental()) >= 0.0, \
    #                 'parameter(s) out of bound, weird: ' \
    #                 + str(pdf_for_ortho_only_optimization.matrix2params_incremental())
    #             assert max(pdf_for_ortho_only_optimization.matrix2params_incremental()) <= 1.0, \
    #                 'parameter(s) out of bound, weird' \
    #                 + str(pdf_for_ortho_only_optimization.matrix2params_incremental())
    #
    #             def cost_function_orthogonal_part(free_params, static_params, entropy_cost_factor=entropy_cost_factor):
    #                 # note: keep entropy_cost_factor < 1 or (close to) 0.0
    #
    #                 assert len(free_params) == num_free_parameters_ortho_only
    #
    #                 pdf_for_ortho_only_optimization.params2matrix_incremental(list(static_params) + list(free_params))
    #
    #                 # also try to minimize the total entropy of the orthogonal variable set, i.e., try to make the
    #                 # orthogonal part 'efficient' in the sense that it uses only as much entropy as it needs to do its
    #                 # job but no more
    #                 if entropy_cost_factor != 0.0:
    #                     entropy_cost = entropy_cost_factor * pdf_for_ortho_only_optimization.entropy(orthogonal_variables)
    #                 else:
    #                     entropy_cost = 0.0  # do not compute entropy if not used anyway
    #
    #                 # MI with 'remaining_variables' is unwanted, but MI with 'variables_to_orthogonalize' is wanted
    #                 cost_ortho = pdf_for_ortho_only_optimization.mutual_information(remaining_variables,
    #                                                                                 orthogonal_variables) \
    #                              - pdf_for_ortho_only_optimization.mutual_information(variables_to_orthogonalize,
    #                                                                                   orthogonal_variables) \
    #                              + entropy_cost
    #
    #                 return float(cost_ortho)
    #
    #             # if verbose and __debug__:
    #             #     static_param_values = self.matrix2params_incremental()
    #             #     free_param_values = pdf_for_ortho_only_optimization.matrix2params_incremental()[len(self.matrix2params_incremental()):]
    #             #
    #             #     print 'debug: append_orthogonalized_variables: orthogonal cost value before optimization =', \
    #             #         cost_function_orthogonal_part(free_param_values, static_param_values), \
    #             #         '(minimum=' + str(-self.entropy(variables_to_orthogonalize)) + ')'
    #
    #             pdf_result.append_optimized_variables(num_added_variables_orthogonal, cost_function_orthogonal_part,
    #                                                   initial_guess=num_repeats)
    #
    #             # why would a parameter already be out of bound
    #             # even if by a small amount)? should be detected already at some earlier stage.
    #             # append_optimized_variables should already fix this itself.
    #             assert min(pdf_result.matrix2params_incremental()) >= 0.0, \
    #                 'parameter(s) out of bound, weird: ' \
    #                 + str(pdf_result.matrix2params_incremental())
    #             assert max(pdf_result.matrix2params_incremental()) <= 1.0, \
    #                 'parameter(s) out of bound, weird' \
    #                 + str(pdf_result.matrix2params_incremental())
    #
    #             # if verbose and __debug__:
    #             #     static_param_values = self.matrix2params_incremental()
    #             #     free_param_values = pdf_result.matrix2params_incremental()[len(self.matrix2params_incremental()):]
    #             #
    #             #     # test whether the 'static' parameters were indeed kept static during optimization
    #             #     np.testing.assert_array_almost_equal(self.matrix2params_incremental(),
    #             #                                          pdf_result.matrix2params_incremental()[:len(self.matrix2params_incremental())])
    #             #
    #             #     print 'debug: append_orthogonalized_variables: orthogonal cost value after optimization =', \
    #             #         cost_function_orthogonal_part(free_param_values, static_param_values)
    #
    #         ### now add the PARALLEL part
    #
    #         if num_added_variables_parallel > 0:
    #             if num_added_variables_orthogonal == 0:
    #                 raise UserWarning('it is ill-defined to add \'parallel\' variables if I do not have any '
    #                                   '\'orthogonal\' variables to minimize MI against. Just also ask for '
    #                                   'orthogonal variables and then remove them (marginalize all other variables)?')
    #
    #             pdf_for_para_only_optimization = pdf_for_ortho_only_optimization.copy()
    #             # todo: it should be possible to let the parallel variables depend only on the orthogonal_variables
    #             # and the variables_to_orthogonalize, not also the remaining_variables, which would greatly
    #             # reduce the number of free parameters. But then you need to add this artial conditional pdf
    #             # to the complete pdf afterward, repeating it in some way.
    #             pdf_for_para_only_optimization.append_variables(num_added_variables_parallel)
    #
    #             num_free_parameters_para_only = len(pdf_for_para_only_optimization.matrix2params_incremental()) \
    #                                             - len(pdf_for_ortho_only_optimization.matrix2params_incremental())
    #
    #             assert num_free_parameters_para_only > 0 or num_added_variables_parallel == 0
    #
    #             parallel_variables = range(len(pdf_for_ortho_only_optimization),
    #                                        len(pdf_for_para_only_optimization))
    #
    #             assert len(np.intersect1d(parallel_variables, orthogonal_variables)) == 0
    #             assert len(np.intersect1d(parallel_variables, remaining_variables)) == 0
    #
    #             assert len(parallel_variables) == num_added_variables_parallel
    #
    #             # due to the scipy's minimize() procedure the list of parameters can be temporarily slightly
    #             # out of bounds, like
    #             # 1.000000001, but soon after this should be clipped to the allowed range and e.g. at this
    #             # point the parameters
    #             # are expected to be all valid
    #             assert min(pdf_for_para_only_optimization.matrix2params_incremental()) >= 0.0, \
    #                 'parameter(s) out of bound: ' + str(pdf_for_para_only_optimization.matrix2params_incremental())
    #             assert max(pdf_for_para_only_optimization.matrix2params_incremental()) <= 1.0, \
    #                 'parameter(s) out of bound: ' + str(pdf_for_para_only_optimization.matrix2params_incremental())
    #
    #             def cost_function_parallel_part(free_params, static_params, entropy_cost_factor=entropy_cost_factor):
    #                 # note: keep entropy_cost_factor < 1 or (close to) 0.0
    #
    #                 assert len(free_params) == num_free_parameters_para_only
    #                 assert len(free_params) > 0, 'makes no sense to optimize 0 parameters'
    #
    #                 pdf_for_para_only_optimization.params2matrix_incremental(list(static_params) + list(free_params))
    #
    #                 # also try to minimize the total entropy of the parallel variable set, i.e., try to make the
    #                 # parallel part 'efficient' in the sense that it uses only as much entropy as it needs to do its
    #                 # job but no more
    #                 if entropy_cost_factor != 0.0:
    #                     entropy_cost = entropy_cost_factor * pdf_for_para_only_optimization.entropy(parallel_variables)
    #                 else:
    #                     entropy_cost = 0.0  # do not compute entropy if not used anyway
    #
    #                 # MI with 'variables_to_orthogonalize' is wanted, but MI with 'orthogonal_variables' is unwanted
    #                 cost_para = - pdf_for_para_only_optimization.mutual_information(variables_to_orthogonalize,
    #                                                                                 parallel_variables) \
    #                             + pdf_for_para_only_optimization.mutual_information(parallel_variables,
    #                                                                                  orthogonal_variables) \
    #                             + entropy_cost
    #
    #                 return float(cost_para)
    #
    #             if verbose and __debug__:
    #                 static_param_values = pdf_for_ortho_only_optimization.matrix2params_incremental()
    #                 free_param_values = pdf_for_para_only_optimization.matrix2params_incremental()[len(pdf_for_ortho_only_optimization.matrix2params_incremental()):]
    #
    #                 print 'debug: append_orthogonalized_variables: parallel cost value before optimization =', \
    #                     cost_function_parallel_part(free_param_values, static_param_values), \
    #                     '(minimum=' + str(-self.entropy(variables_to_orthogonalize)) + ')'
    #
    #                 # note: this is a probabilistic check: with high probability the 0.0 value is suspected to
    #                 # lead to the asserted condition, but it is also possible that 0.0 just so happens to be due to
    #                 # randomness, however this should be with very small probability
    #                 if cost_function_parallel_part(free_param_values, static_param_values) == 0.0:
    #                     # is this because all entropy of the <variables_to_orthogonalize> is already completely
    #                     # in <orthogonal_variables>, so that for <parallel_variables> there is no more entropy left?
    #                     try:
    #                         np.testing.assert_almost_equal(
    #                             pdf_for_para_only_optimization.mutual_information(remaining_variables, orthogonal_variables)
    #                             ,
    #                             pdf_for_para_only_optimization.entropy(variables_to_orthogonalize)
    #                         )
    #                     except AssertionError as e:
    #                         print 'error: pdf_for_para_only_optimization.' \
    #                               'mutual_information(remaining_variables, orthogonal_variables) =', \
    #                             pdf_for_para_only_optimization.mutual_information(remaining_variables,
    #                                                                               orthogonal_variables)
    #                         print 'error: pdf_for_para_only_optimization.entropy(variables_to_orthogonalize) =', \
    #                             pdf_for_para_only_optimization.entropy(variables_to_orthogonalize)
    #                         print 'error: pdf_for_para_only_optimization.' \
    #                               'mutual_information(parallel_variables, orthogonal_variables) =', \
    #                             pdf_for_para_only_optimization.mutual_information(parallel_variables,
    #                                                                                orthogonal_variables)
    #                         print 'error: pdf_for_para_only_optimization.' \
    #                               'mutual_information(variables_to_orthogonalize, parallel_variables) =', \
    #                             pdf_for_para_only_optimization.mutual_information(variables_to_orthogonalize,
    #                                                                               parallel_variables)
    #                         print 'error: pdf_for_para_only_optimization.entropy(remaining_variables)', \
    #                             pdf_for_para_only_optimization.entropy(remaining_variables)
    #                         print 'error: pdf_for_para_only_optimization.entropy(orthogonal_variables) =', \
    #                             pdf_for_para_only_optimization.entropy(orthogonal_variables)
    #                         print 'error: pdf_for_para_only_optimization.entropy(parallel_variables) =', \
    #                             pdf_for_para_only_optimization.entropy(parallel_variables)
    #
    #                         raise AssertionError(e)
    #
    #             pdf_result.append_optimized_variables(num_added_variables_parallel, cost_function_parallel_part)
    #
    #             # due to the scipy's minimize() procedure the list of parameters can be temporarily slightly
    #             # out of bounds, like
    #             # 1.000000001, but soon after this should be clipped to the allowed range and e.g. at this
    #             # point the parameters
    #             # are expected to be all valid
    #             assert min(pdf_result.matrix2params_incremental()) >= 0.0, \
    #                 'parameter(s) out of bound: ' + str(pdf_result.matrix2params_incremental())
    #             assert max(pdf_result.matrix2params_incremental()) <= 1.0, \
    #                 'parameter(s) out of bound: ' + str(pdf_result.matrix2params_incremental())
    #
    #             if verbose and __debug__:
    #                 static_param_values = pdf_for_ortho_only_optimization.matrix2params_incremental()
    #                 free_param_values = pdf_result.matrix2params_incremental()[len(pdf_for_ortho_only_optimization.matrix2params_incremental()):]
    #
    #                 print 'debug: append_orthogonalized_variables: parallel cost value after optimization =', \
    #                     cost_function_parallel_part(free_param_values, static_param_values), \
    #                     '(minimum=' + str(-self.entropy(variables_to_orthogonalize)) + ')'
    #
    #                 # note: this is a probabilistic check: with high probability the 0.0 value is suspected to
    #                 # lead to the asserted condition, but it is also possible that 0.0 just so happens to be due to
    #                 # randomness, however this should be with very small probability
    #                 if cost_function_parallel_part(free_param_values, static_param_values) == 0.0:
    #                     # is this because all entropy of the <variables_to_orthogonalize> is already completely
    #                     # in <orthogonal_variables>, so that for <parallel_variables> there is no more entropy left?
    #                     np.testing.assert_almost_equal(
    #                         pdf_for_para_only_optimization.mutual_information(remaining_variables, orthogonal_variables)
    #                         ,
    #                         pdf_for_para_only_optimization.entropy(variables_to_orthogonalize)
    #                     )
    #
    #         self.duplicate(pdf_result)
    #
    #
    #
    #     # todo: add some debug tolerance measures to detect if the error is too large in orthogonalization, like >10%
    #     # of entropy of orthogonal_variables is MI with parallel_variables?


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

        for at_level in xrange(1, max_level + 1):
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

    def imbalanced_tree_from_scalars(self, list_of_scalars, numvalues):
        """
        Helper function.
        Consider e.g. tree =
                        [0.36227870614214747,
                         0.48474422004766832,
                         [0.34019329926554265,
                          0.40787146599658614,
                          [0.11638879037422999, 0.64823088842780996],
                          [0.33155311703042312, 0.11398958845340294],
                          [0.13824154613818085, 0.42816388506114755]],
                         [0.15806602176772611,
                          0.32551465875945773,
                          [0.25748947995256499, 0.35415524846620511],
                          [0.64896559115417218, 0.65575802084978507],
                          [0.36051945555508391, 0.40134903827671109]],
                         [0.40568439663760192,
                          0.67602830725264651,
                          [0.35103999983495449, 0.59577145940649334],
                          [0.38917741342947187, 0.44327101890582132],
                          [0.075034425516081762, 0.59660319391007388]]]

        If you first call scalars_up_to_level on this you get a list [0.36227870614214747, 0.48474422004766832,
        0.34019329926554265, 0.40787146599658614, 0.15806602176772611, ...]. If you pass this flattened list through
        this function then you should get the above imbalanced tree structure back again.

        At each level in the resulting tree there will be <numvalues-1> scalars and <numvalues> subtrees (lists).
        :type list_of_scalars: list
        :type numvalues: int
        :rtype: list
        """

        num_levels = int(np.round(np.log2(len(list_of_scalars) + 1) / np.log2(numvalues)))

        all_scalars_at_level = dict()

        list_of_scalars_remaining = list(list_of_scalars)

        for level in xrange(num_levels):
            num_scalars_at_level = np.power(numvalues, level) * (numvalues - 1)

            scalars_at_level = list_of_scalars_remaining[:num_scalars_at_level]

            all_scalars_at_level[level] = scalars_at_level

            list_of_scalars_remaining = list_of_scalars_remaining[num_scalars_at_level:]

        def tree_from_levels(all_scalars_at_level):
            if len(all_scalars_at_level) == 0:
                return []
            else:
                assert len(all_scalars_at_level[0]) == numvalues - 1

                if len(all_scalars_at_level) > 1:
                    assert len(all_scalars_at_level[1]) == numvalues * (numvalues - 1)
                if len(all_scalars_at_level) > 2:
                    assert len(all_scalars_at_level[2]) == (numvalues*numvalues) * (numvalues - 1), \
                        'len(all_scalars_at_level[2]) = ' + str(len(all_scalars_at_level[2])) + ', ' \
                        '(numvalues*numvalues) * (numvalues - 1) = ' + str((numvalues*numvalues) * (numvalues - 1))
                if len(all_scalars_at_level) > 3:
                    assert len(all_scalars_at_level[3]) == (numvalues*numvalues*numvalues) * (numvalues - 1)
                # etc.

                tree = list(all_scalars_at_level[0][:(numvalues - 1)])

                if len(all_scalars_at_level) > 1:
                    # add <numvalues> subtrees to this level
                    for subtree_id in xrange(numvalues):
                        all_scalars_for_subtree = dict()

                        for level in xrange(len(all_scalars_at_level) - 1):
                            num_scalars_at_level = len(all_scalars_at_level[level + 1])

                            assert np.mod(num_scalars_at_level, numvalues) == 0, 'should be divisible nu <numvalues>'

                            num_scalars_for_subtree = int(num_scalars_at_level / numvalues)

                            all_scalars_for_subtree[level] = \
                                all_scalars_at_level[level + 1][subtree_id * num_scalars_for_subtree
                                                                :(subtree_id + 1) * num_scalars_for_subtree]

                        subtree_i = tree_from_levels(all_scalars_for_subtree)

                        if len(all_scalars_for_subtree) > 1:
                            # numvalues - 1 scalars and numvalues subtrees
                            assert len(subtree_i) == (numvalues - 1) + numvalues, 'len(subtree_i) = ' \
                                                                                  + str(len(subtree_i)) \
                                                                                  + ', expected = ' \
                                                                                  + str((numvalues - 1) + numvalues)
                        elif len(all_scalars_for_subtree) == 1:
                            assert len(subtree_i) == numvalues - 1

                        tree.append(subtree_i)

                return tree

        tree = tree_from_levels(all_scalars_at_level)

        assert maximum_depth(tree) == len(all_scalars_at_level)  # should be numvariables if the scalars are parameters
        assert len(flatten(tree)) == len(list_of_scalars), 'all scalars should end up in the tree, and not duplicate'

        return tree


# maybe some day generalize to "ParameterizedJointProbabilityMatrix"...
class TemporalJointProbabilityMatrix(object):

    # this can be taken as Pr(X_{t=0})
    current_pdf = JointProbabilityMatrix(0, 0)  # just so that IDE gets the type of member right

    # this can be taken as Pr(X_{t+1} | X_{t})
    cond_next_pdf = ConditionalProbabilityMatrix()


    def __init__(self, current_pdf, cond_next_pdf):
        """

        :param current_pdf:

        Note: you can get unexpected results if you pass me a PDF object but also use the same objet in other places,
        because ***** Python passes it by reference silently. So I copy it below, which costs memory but saves
        frustration; if you are really sure that you don't get conflicts of a shared object and need memory space,
        you could remove the .copy().
        :type current_pdf: JointProbabilityMatrix
        :param cond_next_pdf: can also be 'random'
        :type cond_next_pdf: ConditionalProbabilityMatrix or str or dict
        """
        self.reset(current_pdf.copy(), cond_next_pdf)


    def reset(self, current_pdf, cond_next_pdf='random'):

        assert current_pdf.numvalues >= 0, 'just for making sure that it seems to be a JointProbabilityMatrix-like ' \
                                           'object'
        assert current_pdf.numvariables >= 0, 'just for making sure that it seems to be a ' \
                                              'JointProbabilityMatrix-like object'

        self.current_pdf = current_pdf

        if cond_next_pdf is None:
            self.cond_next_pdf = ConditionalProbabilityMatrix({states: current_pdf
                                                               for states in current_pdf.statespace()})
        elif cond_next_pdf == 'random':
            self.cond_next_pdf = ConditionalProbabilityMatrix({states: JointProbabilityMatrix(current_pdf.numvariables,
                                                                                              current_pdf.numvalues)
                                                               for states in current_pdf.statespace()})
        elif type(cond_next_pdf) == dict:
            self.cond_next_pdf = ConditionalProbabilityMatrix(cond_next_pdf)
        else:
            assert hasattr(cond_next_pdf, 'iterkeys'), 'should be a ConditionalProbabilityMatrix'
            assert hasattr(cond_next_pdf, 'num_given_variables'), 'should be a ConditionalProbabilityMatrix'

            self.cond_next_pdf = cond_next_pdf


    def __eq__(self, other):

        assert False, 'looking for bug'


    def next(self, current_state=None):  # PDF of next time step

        if current_state is None:
            # todo: for certain cases, like full nested ndarray matrix of probabilities, I think this can be written
            # as a matrix/numpy multiplication, which could be much faster
            new_joint_probs_array = np.sum([self.current_pdf(states)
                                            * self.cond_next_pdf[states].joint_probabilities.joint_probabilities
                                            for states in self.current_pdf.statespace()], axis=0)
        else:
            new_joint_probs_array = self.cond_next_pdf[current_state].joint_probabilities.joint_probabilities

        ret_temp_pdf = self.copy()

        assert np.shape(new_joint_probs_array) == np.shape(ret_temp_pdf.current_pdf.joint_probabilities.joint_probabilities)

        # todo: would be more neat to do this through a reset() or so, but I am missing a constructor now that is
        # flexible, i.e., can also take a np.ndarray or a Nested... object
        ret_temp_pdf.current_pdf.joint_probabilities.reset(new_joint_probs_array)

        assert ret_temp_pdf.current_pdf.joint_probabilities.num_values() == self.current_pdf.joint_probabilities.num_values()
        assert ret_temp_pdf.current_pdf.joint_probabilities.num_variables() == self.current_pdf.joint_probabilities.num_variables()

        return ret_temp_pdf


    def next_conditional(self):
        return copy.deepcopy(self.cond_next_pdf)  # simple enough


    def future(self, numsteps, current_state=None):
        new_pdf = self.copy()

        new_pdf.steps(numsteps, current_state=current_state)

        return new_pdf


    def future_conditional(self, numsteps):

        # if retained_variables != 'all':
        #     retained_variables = tuple(retained_variables)

        if numsteps == 0:
            # pdf will not change regardless of current state
            return ConditionalProbabilityMatrix({states: self.current_pdf
                                                 for states, pdf in self.cond_next_pdf.iteritems()})
        elif numsteps == 1:
            return self.next_conditional()
        elif numsteps > 0:
            return ConditionalProbabilityMatrix({states: TemporalJointProbabilityMatrix(pdf, self.cond_next_pdf).future(numsteps - 1).current_pdf
                                                 for states, pdf in self.cond_next_pdf.iteritems()})
        else:
            raise NotImplementedError('numsteps cannot be < 0.')


    def step(self, current_state=None):
        self.duplicate(self.next(current_state=current_state))


    def steps(self, numsteps, current_state=None):
        if current_state is None:
            for i in xrange(numsteps):
                self.step(current_state=current_state)
        elif numsteps > 0:
            # only constrain the current state for the first step, after that sum over all possible states
            self.step(current_state=current_state)

            for i in xrange(numsteps - 1):
                self.step(current_state=None)
        else:
            pass  # nothing to do


    def duplicate(self, other):
        self.reset(other.current_pdf, other.cond_next_pdf)


    '''
    HOW TO get to IDT picture:

    > temp_pdf = jointpdf.TemporalJointProbabilityMatrix(jointpdf.JointProbabilityMatrix(3, 2), 'random')
    > [plt.plot([temp_pdf.mutual_information([ix], range(len(temp_pdf.current_pdf)), dt2=dt) for dt in range(10)], '-') for ix in range(len(temp_pdf.current_pdf))]; plt.yscale('log'); plt.show()
    '''
    def mutual_information(self, variables1, variables2, dt2=0):
        if dt2 == 0:
            return self.current_pdf.mutual_information(variables1, variables2)
        elif dt2 > 0:
            # cond_pdf_dt = self.future_conditional(dt2,
            #                                       retained_variables=variables2)  # try to save some space
            #
            # joint_pdf = self.current_pdf + cond_pdf_dt
            # assert len(joint_pdf) == len(self.current_pdf) + len(variables2)
            #
            # return joint_pdf.mutual_information(variables1, range(len(self.current_pdf), len(self.current_pdf) + len(variables2)))
            cond_pdf_dt = self.future_conditional(dt2)

            joint_pdf = self.current_pdf + cond_pdf_dt
            assert len(joint_pdf) == len(self.current_pdf) * 2

            return joint_pdf.mutual_information(variables1, np.add(len(self.current_pdf), variables2))


    def copy(self):
        """
        :rtype: TemporalJointProbabilityMatrix
        """
        return copy.deepcopy(self)


### UNIT TESTING:



def run_append_and_marginalize_test():
    pdf = JointProbabilityMatrix(3, 2)

    pdf_copy = pdf.copy()

    pdf.append_variables(4)

    pdf_old = pdf.marginalize_distribution(range(3))

    assert pdf_copy == pdf_old, 'adding and then removing variables should result in the same joint pdf'

    # old_params = pdf_copy.matrix2params_incremental()
    #
    # np.testing.assert_array_almost_equal(pdf.matrix2params_incremental()[:len(old_params)], old_params)


def run_reorder_test():
    pdf = JointProbabilityMatrix(4, 3)

    pdf_original = pdf.copy()

    pdf.reorder_variables([3,2,1,0])
    np.testing.assert_almost_equal(pdf.entropy(), pdf_original.entropy())
    assert pdf != pdf_original

    pdf.reorder_variables([3,2,1,0,0,1,2,3])
    assert len(pdf) == 2 * len(pdf_original)
    np.testing.assert_almost_equal(pdf.entropy(), pdf_original.entropy())
    # the first 4 and second 4 variables should be identical so MI should be maximum
    np.testing.assert_almost_equal(pdf.mutual_information([0,1,2,3], [4,5,6,7]), pdf_original.entropy())

    assert pdf.marginalize_distribution([0,1,2,3]) == pdf_original


def run_params2matrix_test():
    pdf = JointProbabilityMatrix(3, 2)

    pdf_copy = pdf.copy()

    pdf_copy.params2matrix(pdf.matrix2params())

    assert pdf_copy == pdf, 'computing parameter values from joint pdf and using those to construct a 2nd joint pdf ' \
                            'should result in two equal pdfs.'


def run_vector2matrix_test():
    pdf = JointProbabilityMatrix(3, 2)

    pdf_copy = pdf.copy()

    pdf_copy.vector2matrix(pdf.matrix2vector())

    assert pdf_copy == pdf, 'computing vector from joint pdf and using that to construct a 2nd joint pdf ' \
                            'should result in two equal pdfs.'


def run_conditional_pdf_test():
    pdf = JointProbabilityMatrix(3, 2)

    pdf_marginal_1 = pdf.marginalize_distribution([1])

    assert pdf_marginal_1.numvariables == 1

    pdf_cond_23_given_0 = pdf.conditional_probability_distribution([1], [0])
    pdf_cond_23_given_1 = pdf.conditional_probability_distribution([1], [1])

    assert pdf_cond_23_given_0.numvariables == 2
    assert pdf_cond_23_given_1.numvariables == 2

    prob_000_joint = pdf([0,0,0])
    prob_000_cond = pdf_marginal_1([0]) * pdf_cond_23_given_0([0,0])

    np.testing.assert_almost_equal(prob_000_cond, prob_000_joint)

    pdf_conds_23 = pdf.conditional_probability_distributions([1])

    assert pdf_conds_23[(0,)] == pdf_cond_23_given_0
    assert pdf_conds_23[(1,)] == pdf_cond_23_given_1


def run_append_using_transitions_table_and_marginalize_test():
    pdf = JointProbabilityMatrix(3, 2)

    pdf_copy = pdf.copy()

    lists_of_possible_given_values = [range(pdf.numvalues) for _ in xrange(pdf.numvariables)]

    state_transitions = [list(existing_vars_values) + list([int(np.mod(np.sum(existing_vars_values), pdf.numvalues))])
                         for existing_vars_values in itertools.product(*lists_of_possible_given_values)]

    pdf.append_variables_using_state_transitions_table(state_transitions)

    assert not hasattr(state_transitions, '__call__'), 'append_variables_using_state_transitions_table should not ' \
                                                       'replace the caller\'s variables'

    assert pdf.numvariables == 4, 'one variable should be added'

    pdf_old = pdf.marginalize_distribution(range(pdf_copy.numvariables))

    assert pdf_copy == pdf_old, 'adding and then removing variables should result in the same joint pdf'
    
    pdf_copy.append_variables_using_state_transitions_table(
        state_transitions=lambda vals, mv: [int(np.mod(np.sum(vals), mv))])

    assert pdf_copy == pdf, 'should be two equivalent ways of appending a deterministic variable'


def run_synergistic_variables_test_with_subjects_and_agnostics(num_subject_vars=2):
    num_agnostic_vars = 1

    pdf = JointProbabilityMatrix(num_subject_vars + num_agnostic_vars, 2)

    pdf_orig = pdf.copy()

    assert num_subject_vars > 0

    subject_variables = np.random.choice(range(len(pdf)), num_subject_vars, replace=False)
    agnostic_variables = np.setdiff1d(range(len(pdf)), subject_variables)

    pdf.append_synergistic_variables(1, subject_variables=subject_variables)

    assert pdf.marginalize_distribution(range(len(pdf_orig))) == pdf_orig, \
        'appending synergistic variables changed the pdf'
    
    synergistic_variables = range(len(pdf_orig), len(pdf))
    
    tol_rel_err = 0.2

    # ideally, the former is max and the latter is zero
    assert pdf.mutual_information(subject_variables, synergistic_variables) \
           > sum([pdf.mutual_information([sv], synergistic_variables) for sv in subject_variables]) \
           or pdf.entropy(synergistic_variables) < 0.01
    
    assert pdf.mutual_information(synergistic_variables, agnostic_variables) / pdf.entropy(synergistic_variables) \
           < tol_rel_err


def run_synergistic_variables_test_with_subjects(num_subject_vars=2):
    num_other_vars = 1

    pdf = JointProbabilityMatrix(num_subject_vars + num_other_vars, 2)

    pdf_orig = pdf.copy()

    assert num_subject_vars > 0

    subject_variables = np.random.choice(range(len(pdf)), num_subject_vars, replace=False)

    pdf.append_synergistic_variables(1, subject_variables=subject_variables)

    assert pdf.marginalize_distribution(range(len(pdf_orig))) == pdf_orig, \
        'appending synergistic variables changed the pdf'

    # ideally, the former is max and the latter is zero
    assert pdf.mutual_information(subject_variables, range(len(pdf_orig), len(pdf))) \
           > sum([pdf.mutual_information([sv], range(len(pdf_orig), len(pdf))) for sv in subject_variables])


# todo: make another such function but now use the subject_variables option of append_synerg*
def run_synergistic_variables_test(numvars=2):
    pdf = JointProbabilityMatrix(numvars, 2)

    pdf_syn = pdf.copy()

    assert pdf_syn == pdf

    # initial_guess_summed_modulo = np.random.choice([True, False])
    initial_guess_summed_modulo = False

    pdf_syn.append_synergistic_variables(1, initial_guess_summed_modulo=initial_guess_summed_modulo, verbose=False)

    assert pdf_syn.numvariables == pdf.numvariables + 1

    pdf_old = pdf_syn.marginalize_distribution(range(pdf.numvariables))

    # trying to figure out why I hit the assertion "pdf == pdf_old"
    np.testing.assert_almost_equal(pdf_old.joint_probabilities.sum(), 1.0), 'all probabilities should sum to 1.0'
    np.testing.assert_almost_equal(pdf.joint_probabilities.sum(), 1.0), 'all probabilities should sum to 1.0'

    np.testing.assert_array_almost_equal(pdf.joint_probabilities, pdf_old.joint_probabilities)
    assert pdf == pdf_old, 'adding and then removing variables should result in the same joint pdf'

    parameters_before = pdf.matrix2params_incremental()

    pdf_add_random = pdf.copy()
    pdf_add_random.append_variables(1)

    np.testing.assert_array_almost_equal(pdf_add_random.matrix2params_incremental()[:len(parameters_before)],
                                                parameters_before)
    np.testing.assert_array_almost_equal(pdf_syn.matrix2params_incremental()[:len(parameters_before)],
                                                parameters_before)

    # note: this assert is in principle probabilistic, because who knows what optimization procedure is used and
    # how much it potentially sucks. So see if you hit this more than once, if you hit it at all.
    assert pdf_add_random.synergistic_information_naive(range(pdf.numvariables, pdf_add_random.numvariables),
                                                                range(pdf.numvariables)) <= \
           pdf_syn.synergistic_information_naive(range(pdf.numvariables, pdf_add_random.numvariables),
                                                         range(pdf.numvariables)), 'surely the optimization procedure' \
                                                                                   ' in append_synergistic_variables ' \
                                                                                   'yields a better syn. info. than ' \
                                                                                   'an appended variable with simply ' \
                                                                                   'random interaction parameters?!'

    np.testing.assert_array_almost_equal(pdf_add_random.matrix2params_incremental()[:len(parameters_before)],
                                                parameters_before)
    np.testing.assert_array_almost_equal(pdf_syn.matrix2params_incremental()[:len(parameters_before)],
                                                parameters_before)

    syninfo = pdf_syn.synergistic_information_naive(range(pdf.numvariables, pdf_add_random.numvariables),
                                                    range(pdf.numvariables))

    condents = [pdf.conditional_entropy([varix]) for varix in xrange(len(pdf))]

    assert syninfo <= min(condents), 'this is a derived maximum in Quax 2015, synergy paper, right?'


def run_orthogonalization_test_null_hypothesis(num_subject_vars=2, num_ortho_vars=1, num_para_vars=1, numvals=2,
                                               verbose=True, num_repeats=5, tol_rel_error=0.05):
    """
    This is similar in spirit to run_orthogonalization_test, except now the variables to orthogonalize (X) is already
    known to be completely decomposable into two parts (X1, X2) which are orthogonal and parallel, resp. (See also
    the description of append_orthogonalized_variables().) So in this case I am sure that
    append_orthogonalized_variables should find very good solutions.
    :param tol_rel_error: A float in [0, 1) and preferably close to 0.0.
    :param num_subject_vars:
    :param num_ortho_vars:
    :param num_para_vars:
    :param numvals:
    :param verbose:
    :param num_repeats:
    """
    pdf = JointProbabilityMatrix(num_subject_vars, numvals)

    # note: I add the 'null hypothesis' parallel and orthogonal parts in reversed order compared to what
    # append_orthogonalized_variables returns, otherwise I have to insert a reordering or implement
    # append_redundant_variables for subsets of variables (not too hard but anyway)

    pdf.append_redundant_variables(num_para_vars)

    # the just added variables should be completely redundant
    np.testing.assert_almost_equal(pdf.mutual_information(range(num_subject_vars),
                                                          range(num_subject_vars, num_subject_vars + num_para_vars)),
                                   pdf.entropy(range(num_subject_vars, num_subject_vars + num_para_vars)))

    pdf.append_independent_variables(JointProbabilityMatrix(num_ortho_vars, pdf.numvalues))

    # the just added variables should be completely independent of all previous
    np.testing.assert_almost_equal(pdf.mutual_information(range(num_subject_vars, num_subject_vars + num_para_vars),
                                                          range(num_subject_vars + num_para_vars,
                                                                num_subject_vars + num_para_vars + num_ortho_vars)),
                                   0.0)

    vars_to_orthogonalize = range(num_subject_vars, num_subject_vars + num_para_vars + num_ortho_vars)

    pdf.append_orthogonalized_variables(vars_to_orthogonalize, num_ortho_vars, num_para_vars,
                                        num_repeats=num_repeats)

    assert len(pdf) == num_subject_vars + num_ortho_vars + num_para_vars + num_ortho_vars + num_para_vars

    result_ortho_vars = range(num_subject_vars + num_ortho_vars + num_para_vars,
                              num_subject_vars + num_ortho_vars + num_para_vars + num_ortho_vars)
    result_para_vars = range(num_subject_vars + num_ortho_vars + num_para_vars + num_ortho_vars,
                             num_subject_vars + num_ortho_vars + num_para_vars + num_ortho_vars + num_para_vars)
    subject_vars = range(num_subject_vars)

    if pdf.entropy(result_ortho_vars) != 0.0:
        try:
            assert pdf.mutual_information(subject_vars, result_ortho_vars) / pdf.entropy(result_ortho_vars) \
                   <= tol_rel_error
        except AssertionError as e:

            print 'debug: pdf.mutual_information(subject_vars, result_ortho_vars) =', \
                pdf.mutual_information(subject_vars, result_ortho_vars)
            print 'debug: pdf.entropy(result_ortho_vars) =', pdf.entropy(result_ortho_vars)
            print 'debug: pdf.mutual_information(subject_vars, result_ortho_vars) / pdf.entropy(result_ortho_vars) =', \
                pdf.mutual_information(subject_vars, result_ortho_vars) / pdf.entropy(result_ortho_vars)
            print 'debug: (ideal of previous quantity: 0.0)'
            print 'debug: tol_rel_error =', tol_rel_error

            raise AssertionError(e)

    if pdf.entropy(result_para_vars) != 0.0:
        assert pdf.mutual_information(subject_vars, result_para_vars) \
               / pdf.entropy(result_para_vars) >= 1.0 - tol_rel_error

    if pdf.entropy(result_ortho_vars) != 0.0:
        assert pdf.mutual_information(result_para_vars, result_ortho_vars) \
               / pdf.entropy(result_ortho_vars) <= tol_rel_error

    if pdf.entropy(result_para_vars) != 0.0:
        assert pdf.mutual_information(result_para_vars, result_ortho_vars) \
               / pdf.entropy(result_para_vars) <= tol_rel_error

    if pdf.entropy(vars_to_orthogonalize) != 0.0:
        try:
            assert pdf.mutual_information(vars_to_orthogonalize, list(result_para_vars) + list(result_ortho_vars)) \
                   / pdf.entropy(vars_to_orthogonalize) >= 1.0 - tol_rel_error, \
                'not all entropy of X is accounted for in {X1, X2}, which is of course the purpose of decomposition.'
        except AssertionError as e:
            print 'debug: pdf.mutual_information(vars_to_orthogonalize, ' \
                  'list(result_para_vars) + list(result_ortho_vars)) =', \
                pdf.mutual_information(vars_to_orthogonalize, list(result_para_vars) + list(result_ortho_vars))
            print 'debug: pdf.entropy(vars_to_orthogonalize) =', pdf.entropy(vars_to_orthogonalize)
            print 'debug: pdf.mutual_information(vars_to_orthogonalize, ' \
                  'list(result_para_vars) + list(result_ortho_vars))' \
                   '/ pdf.entropy(vars_to_orthogonalize) =', \
                pdf.mutual_information(vars_to_orthogonalize, list(result_para_vars) + list(result_ortho_vars)) \
                   / pdf.entropy(vars_to_orthogonalize)
            print 'debug: tol_rel_error =', tol_rel_error


def run_orthogonalization_test(num_subject_vars=2, num_orthogonalized_vars=1, numvals=3, verbose=True, num_repeats=1):
    """

    :param num_subject_vars:
    :param num_orthogonalized_vars:
    :param numvals:
    :param verbose:
    :param num_repeats:
    :raise AssertionError:
    """

    # note: in the code I assume that the number of added orthogonal variables equals num_orthogonalized_vars
    # note: in the code I assume that the number of added parallel variables equals num_orthogonalized_vars

    # todo: I am checking if it should be generally expected at all that any given X for any (Y, X) can be decomposed
    # into X1, X2 (orthogonal and parallel, resp.). Maybe in some cases it is not possible, due to the discrete nature
    # of the variables.

    pdf = JointProbabilityMatrix(num_subject_vars + num_orthogonalized_vars, numvals)

    pdf_original = pdf.copy()

    subject_vars = range(num_subject_vars)
    # these are the variables that will be 'orthogonalized', i.e., naming this X then the below appending procedure
    # will find two variable sets {X1,X2}=X where X1 is 'orthogonal' to original_vars (MI=0) and X2 is 'parallel'
    vars_to_orthogonalize = range(num_subject_vars, num_subject_vars + num_orthogonalized_vars)

    pdf.append_orthogonalized_variables(vars_to_orthogonalize, num_orthogonalized_vars, num_orthogonalized_vars,
                                        num_repeats=num_repeats)

    if verbose:
        print 'debug: computed first orthogonalization'

    assert len(pdf) == len(pdf_original) + num_orthogonalized_vars * 2

    # note: implicitly, the number of variables added for the orthogonal part is <num_orthogonalized_vars> here
    ortho_vars = range(num_subject_vars + num_orthogonalized_vars,
                       num_subject_vars + num_orthogonalized_vars + num_orthogonalized_vars)
    # note: implicitly, the number of variables added for the parallel part is <num_orthogonalized_vars> here
    para_vars = range(num_subject_vars + num_orthogonalized_vars + num_orthogonalized_vars,
                      num_subject_vars + num_orthogonalized_vars + num_orthogonalized_vars + num_orthogonalized_vars)

    assert para_vars[-1] == len(pdf) - 1, 'not all variables accounted for?'

    '''
    these should be high:
    pdf.mutual_information(original_vars, para_vars)
    pdf.mutual_information(vars_to_orthogonalize, ortho_vars)

    these should be low:
    pdf.mutual_information(original_vars, ortho_vars)
    pdf.mutual_information(para_vars, ortho_vars)
    '''

    tol_rel_err = 0.2  # used for all kinds of tests

    # some test of ordering.
    # Here: more % of the parallel variables' entropy should be correlated with subject_variables
    # than the same % for the orthogonal variables. This should be a pretty weak requirement.
    assert pdf.mutual_information(subject_vars, para_vars) / pdf.entropy(para_vars) \
           > pdf.mutual_information(subject_vars, ortho_vars) / pdf.entropy(ortho_vars), \
        '1: pdf.mutual_information(vars_to_orthogonalize, para_vars) = ' + str(pdf.mutual_information(vars_to_orthogonalize, para_vars)) \
        + ', pdf.mutual_information(para_vars, ortho_vars) = ' + str(pdf.mutual_information(para_vars, ortho_vars)) \
        + ', vars_to_orthogonalize=' + str(vars_to_orthogonalize) + ', para_vars=' + str(para_vars) \
        + ', ortho_vars=' + str(ortho_vars) + ', subject_vars=' + str(subject_vars) \
        + ', pdf.mutual_information(subject_vars, para_vars) = ' + str(pdf.mutual_information(subject_vars, para_vars)) \
        + ', pdf.mutual_information(subject_vars, ortho_vars) = ' + str(pdf.mutual_information(subject_vars, ortho_vars)) \
        + ', pdf.entropy(subject_vars) = ' + str(pdf.entropy(subject_vars)) \
        + ', pdf.entropy(vars_to_orthogonalize) = ' + str(pdf.entropy(vars_to_orthogonalize)) \
        + ', pdf.entropy(ortho_vars) = ' + str(pdf.entropy(ortho_vars)) \
        + ', pdf.entropy(para_vars) = ' + str(pdf.entropy(para_vars)) \
        + ', pdf.mutual_information(vars_to_orthogonalize, ortho_vars) = ' \
        + str(pdf.mutual_information(vars_to_orthogonalize, ortho_vars))

    assert pdf.mutual_information(vars_to_orthogonalize, ortho_vars) > pdf.mutual_information(para_vars, ortho_vars), \
        '2: pdf.mutual_information(vars_to_orthogonalize, para_vars) = ' + str(pdf.mutual_information(vars_to_orthogonalize, para_vars)) \
        + ', pdf.mutual_information(para_vars, ortho_vars) = ' + str(pdf.mutual_information(para_vars, ortho_vars)) \
        + ', vars_to_orthogonalize=' + str(vars_to_orthogonalize) + ', para_vars=' + str(para_vars) \
        + ', ortho_vars=' + str(ortho_vars) + ', subject_vars=' + str(subject_vars) \
        + ', pdf.mutual_information(subject_vars, para_vars) = ' + str(pdf.mutual_information(subject_vars, para_vars)) \
        + ', pdf.mutual_information(subject_vars, ortho_vars) = ' + str(pdf.mutual_information(subject_vars, ortho_vars)) \
        + ', pdf.entropy(subject_vars) = ' + str(pdf.entropy(subject_vars)) \
        + ', pdf.entropy(vars_to_orthogonalize) = ' + str(pdf.entropy(vars_to_orthogonalize)) \
        + ', pdf.entropy(ortho_vars) = ' + str(pdf.entropy(ortho_vars)) \
        + ', pdf.entropy(para_vars) = ' + str(pdf.entropy(para_vars)) \
        + ', pdf.mutual_information(vars_to_orthogonalize, ortho_vars) = ' \
        + str(pdf.mutual_information(vars_to_orthogonalize, ortho_vars))

    assert pdf.mutual_information(vars_to_orthogonalize, para_vars) > pdf.mutual_information(para_vars, ortho_vars), \
        '3: pdf.mutual_information(vars_to_orthogonalize, para_vars) = ' + str(pdf.mutual_information(vars_to_orthogonalize, para_vars)) \
        + ', pdf.mutual_information(para_vars, ortho_vars) = ' + str(pdf.mutual_information(para_vars, ortho_vars)) \
        + ', vars_to_orthogonalize=' + str(vars_to_orthogonalize) + ', para_vars=' + str(para_vars) \
        + ', ortho_vars=' + str(ortho_vars) + ', subject_vars=' + str(subject_vars) \
        + ', pdf.mutual_information(subject_vars, para_vars) = ' + str(pdf.mutual_information(subject_vars, para_vars)) \
        + ', pdf.mutual_information(subject_vars, ortho_vars) = ' + str(pdf.mutual_information(subject_vars, ortho_vars)) \
        + ', pdf.entropy(subject_vars) = ' + str(pdf.entropy(subject_vars)) \
        + ', pdf.entropy(vars_to_orthogonalize) = ' + str(pdf.entropy(vars_to_orthogonalize)) \
        + ', pdf.entropy(ortho_vars) = ' + str(pdf.entropy(ortho_vars)) \
        + ', pdf.entropy(para_vars) = ' + str(pdf.entropy(para_vars)) \
        + ', pdf.mutual_information(vars_to_orthogonalize, ortho_vars) = ' \
        + str(pdf.mutual_information(vars_to_orthogonalize, ortho_vars))

    # at least more than <tol*>% of the entropy of para_vars is not 'wrong' (correlated with ortho_vars)
    assert pdf.mutual_information(para_vars, ortho_vars) < tol_rel_err * pdf.entropy(para_vars)
    # at least more than <tol*>% of the entropy of ortho_vars is not 'wrong' (correlated with para_vars)
    assert pdf.mutual_information(para_vars, ortho_vars) < tol_rel_err * pdf.entropy(ortho_vars)

    # todo: print some numbers to get a feeling of what (verbose) kind of accuracy I get, or let
    # append_orthogonalized_variables return those (e.g. in a dict?)?

    if verbose:
        print 'debug: computed first bunch of MI checks'

    ### test if the total entropy of ortho_vars + para_vars is close enough to the entropy of vars_to_orthogonalize

    # note: cannot directly use pdf.entropy(ortho_vars + para_vars) because I did not yet include a cost term
    # for the total entropy ('efficiency') of the ortho and para vars (so far entropy_cost_factor=0.0)
    entropy_ortho_and_para = pdf.mutual_information(vars_to_orthogonalize, ortho_vars + para_vars)
    entropy_vars_to_ortho = pdf.entropy(vars_to_orthogonalize)

    if verbose:
        print 'debug: computed few more entropy things'

    if entropy_vars_to_ortho != 0.0:
        assert abs(entropy_vars_to_ortho - entropy_ortho_and_para) / entropy_vars_to_ortho <= tol_rel_err, \
            'the total entropy of the ortho and para vars is too high/low: ' + str(entropy_ortho_and_para) + ' versus' \
            + ' entropy_vars_to_ortho=' + str(entropy_vars_to_ortho) + ' (rel. err. tol. = ' + str(tol_rel_err) + ')'

    ideal_para_entropy = pdf_original.mutual_information(vars_to_orthogonalize, subject_vars)

    assert pdf.entropy(para_vars) >= (1.0 - tol_rel_err) * ideal_para_entropy
    try:
        assert pdf.mutual_information(vars_to_orthogonalize, para_vars) >= (1.0 - tol_rel_err) * ideal_para_entropy
    except AssertionError as e:
        print 'debug: pdf.mutual_information(vars_to_orthogonalize, para_vars) =', \
            pdf.mutual_information(vars_to_orthogonalize, para_vars)
        print 'debug: pdf.mutual_information(subject_vars, para_vars) =', \
            pdf.mutual_information(subject_vars, para_vars)
        print 'debug: ideal_para_entropy =', ideal_para_entropy
        print 'debug: tol_rel_err =', tol_rel_err

        raise AssertionError(e)

    if verbose:
        print 'debug: ...and more...'

    ideal_ortho_entropy = pdf_original.conditional_entropy(vars_to_orthogonalize)

    # H(X1) should be close to H(X|Y), i.e., the entropy in X which does not share information with Y
    if not pdf.entropy(ortho_vars) == 0.0:
        assert abs(pdf.entropy(ortho_vars) - ideal_ortho_entropy) / pdf.entropy(ortho_vars) <= tol_rel_err

    # I(X1:X) should be (almost) equal to H(X1), i.e., all entropy of the orthogonal X1 should be from X, nowhere
    # else
    if not pdf.mutual_information(vars_to_orthogonalize, ortho_vars) == 0.0:
        assert abs(pdf.mutual_information(vars_to_orthogonalize, ortho_vars) - ideal_ortho_entropy) \
               / pdf.mutual_information(vars_to_orthogonalize, ortho_vars) <= tol_rel_err, \
        'pdf.mutual_information(vars_to_orthogonalize, para_vars) = ' + str(pdf.mutual_information(vars_to_orthogonalize, para_vars)) \
        + ', pdf.mutual_information(para_vars, ortho_vars) = ' + str(pdf.mutual_information(para_vars, ortho_vars)) \
        + ', vars_to_orthogonalize=' + str(vars_to_orthogonalize) + ', para_vars=' + str(para_vars) \
        + ', ortho_vars=' + str(ortho_vars) + ', subject_vars=' + str(subject_vars) \
        + ', pdf.mutual_information(subject_vars, para_vars) = ' + str(pdf.mutual_information(subject_vars, para_vars)) \
        + ', pdf.mutual_information(subject_vars, ortho_vars) = ' + str(pdf.mutual_information(subject_vars, ortho_vars)) \
        + ', pdf.entropy(subject_vars) = ' + str(pdf.entropy(subject_vars)) \
        + ', pdf.entropy(vars_to_orthogonalize) = ' + str(pdf.entropy(vars_to_orthogonalize)) \
        + ', pdf.entropy(ortho_vars) = ' + str(pdf.entropy(ortho_vars)) \
        + ', pdf.entropy(para_vars) = ' + str(pdf.entropy(para_vars)) \
        + ', pdf.mutual_information(vars_to_orthogonalize, ortho_vars) = ' \
        + str(pdf.mutual_information(vars_to_orthogonalize, ortho_vars))

    if verbose:
        print 'debug: done!'



def run_append_conditional_pdf_test():
    pdf_joint = JointProbabilityMatrix(4, 3)

    pdf_12 = pdf_joint.marginalize_distribution([0, 1])
    pdf_34_cond_12 = pdf_joint.conditional_probability_distributions([0, 1])

    pdf_merged = pdf_12.copy()
    pdf_merged.append_variables_using_conditional_distributions(pdf_34_cond_12)

    assert pdf_merged == pdf_joint

    ### add a single variable conditioned on only the second existing variable

    rand_partial_cond_pdf = {(val,): JointProbabilityMatrix(1, pdf_joint.numvalues)
                             for val in xrange(pdf_joint.numvalues)}

    pdf_orig = pdf_joint.copy()

    pdf_joint.append_variables_using_conditional_distributions(rand_partial_cond_pdf, [1])

    assert pdf_orig == pdf_joint.marginalize_distribution(range(len(pdf_orig)))

    assert pdf_joint.mutual_information([0], [len(pdf_joint) - 1]) < 0.01, 'should be close to 0, not conditioned on'
    assert pdf_joint.mutual_information([1], [len(pdf_joint) - 1]) > 0.0, 'should not be 0, it is conditioned on'
    assert pdf_joint.mutual_information([2], [len(pdf_joint) - 1]) < 0.01, 'should be close to 0, not conditioned on'
    assert pdf_joint.mutual_information([3], [len(pdf_joint) - 1]) < 0.01, 'should be close to 0, not conditioned on'


def run_append_conditional_entropy_test():
    pdf_joint = JointProbabilityMatrix(4, 3)

    assert pdf_joint.conditional_entropy([1,2]) >= pdf_joint.conditional_entropy([1])
    assert pdf_joint.conditional_entropy([1,0]) >= pdf_joint.conditional_entropy([1])

    assert pdf_joint.conditional_entropy([0,2]) <= pdf_joint.entropy([0,2])

    assert pdf_joint.entropy([]) == 0, 'H(<empty-set>)=0 ... right? Yes think so'

    np.testing.assert_almost_equal(pdf_joint.conditional_entropy([1,2], [1,2]), 0.0)
    np.testing.assert_almost_equal(pdf_joint.entropy([0]) + pdf_joint.conditional_entropy([1,2,3], [0]),
                                   pdf_joint.entropy([0,1,2,3]))
    np.testing.assert_almost_equal(pdf_joint.conditional_entropy([0,1,2,3]), pdf_joint.entropy())


def run_params2matrix_incremental_test(numvars=3):
    pdf1 = JointProbabilityMatrix(numvars, 3)
    pdf2 = JointProbabilityMatrix(numvars, 3)

    params1 = pdf1.matrix2params_incremental(return_flattened=True)
    tree1 = pdf1.matrix2params_incremental(return_flattened=False)
    tree11 = pdf1.imbalanced_tree_from_scalars(params1, pdf1.numvalues)
    params1_from_tree1 = pdf1.scalars_up_to_level(tree1)
    params1_from_tree11 = pdf1.scalars_up_to_level(tree11)

    np.testing.assert_array_almost_equal(params1, params1_from_tree11)  # more a test of tree conversion itself
    np.testing.assert_array_almost_equal(params1, params1_from_tree1)

    pdf2.params2matrix_incremental(params1)

    params2 = pdf2.matrix2params_incremental()

    assert pdf1 == pdf2, 'computing parameter values from joint pdf and using those to construct a 2nd joint pdf ' \
                         'should result in two equal pdfs.\nparams1 = ' + str(params1) + '\nparms2 = ' + str(params2)

    pdf2.params2matrix_incremental(pdf2.matrix2params_incremental())

    assert pdf1 == pdf2, 'computing parameter values from joint pdf and using those to reconstruct the joint pdf ' \
                         'should result in two equal pdfs.'

    ### TEST the incrementality of the parameters

    pdf_marginal = pdf1.marginalize_distribution([0])
    params_marginal = pdf_marginal.matrix2params_incremental()
    np.testing.assert_array_almost_equal(params_marginal, pdf1.matrix2params_incremental()[:len(params_marginal)])

    pdf_marginal = pdf1.marginalize_distribution([0, 1])
    params_marginal = pdf_marginal.matrix2params_incremental()
    try:
        np.testing.assert_array_almost_equal(flatten(params_marginal),
                                             flatten(pdf1.matrix2params_incremental()[:len(params_marginal)]))
    except AssertionError as e:
        print '---------------------'
        print 'debug: params_marginal =                 ', np.round(params_marginal, decimals=4)
        print 'debug: pdf1.matrix2params_incremental() =', np.round(pdf1.matrix2params_incremental(), 4)
        print '---------------------'
        print 'debug: params_marginal =                 ', \
            pdf_marginal.matrix2params_incremental(return_flattened=False)
        print 'debug: pdf1.matrix2params_incremental() =', \
            pdf1.matrix2params_incremental(return_flattened=False)
        print '---------------------'

        raise AssertionError(e)

    if numvars >= 3:
        pdf_marginal = pdf1.marginalize_distribution([0, 1, 2])
        params_marginal = pdf_marginal.matrix2params_incremental()
        np.testing.assert_array_almost_equal(flatten(params_marginal),
                                             flatten(pdf1.matrix2params_incremental()[:len(params_marginal)]))


def run_scalars_to_tree_test():
    pdf = JointProbabilityMatrix(4, 3)

    list_of_scalars = pdf.matrix2params()  # does not matter what sequence of numbers, as long as the length is correct

    tree = pdf.imbalanced_tree_from_scalars(list_of_scalars, pdf.numvalues)
    np.testing.assert_array_almost_equal(pdf.scalars_up_to_level(tree), list_of_scalars)
    # np.testing.assert_array_almost_equal(pdf.scalars_up_to_level(tree), pdf.matrix2params_incremental())

    # another tree
    tree = pdf.matrix2params_incremental(return_flattened=False)

    # assert not np.isscalar(tree[-1]), 'you changed the matrix2params_incremental to return flat lists, which is good ' \
    #                                   'but then you should change this test, set an argument like flatten=False?'

    list_of_scalars2 = pdf.scalars_up_to_level(tree)

    tree2 = pdf.imbalanced_tree_from_scalars(list_of_scalars2, pdf.numvalues)

    np.testing.assert_array_almost_equal(flatten(tree), flatten(tree2))
    np.testing.assert_array_almost_equal(pdf.scalars_up_to_level(tree), pdf.scalars_up_to_level(tree2))


def run_all_tests(verbose=True, all_inclusive=False):
    run_append_and_marginalize_test()
    if verbose:
        print 'note: test run_append_and_marginalize_test successful.'

    run_params2matrix_test()
    if verbose:
        print 'note: test run_params2matrix_test successful.'

    run_vector2matrix_test()
    if verbose:
        print 'note: test run_vector2matrix_test successful.'

    run_conditional_pdf_test()
    if verbose:
        print 'note: test run_conditional_pdf_test successful.'

    run_append_using_transitions_table_and_marginalize_test()
    if verbose:
        print 'note: test run_append_using_transitions_table_and_marginalize_test successful.'

    run_append_conditional_pdf_test()
    if verbose:
        print 'note: test run_append_conditional_pdf_test successful.'

    run_scalars_to_tree_test()
    if verbose:
        print 'note: test run_scalars_to_tree_test successful.'

    run_params2matrix_incremental_test()
    if verbose:
        print 'note: test run_params2matrix_incremental_test successful.'

    run_synergistic_variables_test()
    if verbose:
        print 'note: test run_synergistic_variables_test successful.'

    run_synergistic_variables_test_with_subjects()
    if verbose:
        print 'note: test run_synergistic_variables_test_with_subjects successful.'

    run_synergistic_variables_test_with_subjects_and_agnostics()
    if verbose:
        print 'note: test run_synergistic_variables_test_with_subjects_and_agnostics successful.'

    run_append_conditional_entropy_test()
    if verbose:
        print 'note: test run_append_conditional_entropy_test successful.'

    run_reorder_test()
    if verbose:
        print 'note: test run_reorder_test successful.'

    if all_inclusive:
        run_orthogonalization_test()
        if verbose:
            print 'note: test run_orthogonalization_test successful.'

    run_orthogonalization_test_null_hypothesis()
    if verbose:
        print 'note: test run_orthogonalization_test_null_hypothesis successful.'

    if verbose:
        print 'note: finished. all tests successful.'


# def sum_modulo(values, modulo):  # deprecated, can be removed?
#     """
#     An example function which can be passed as the state_transitions argument to the
#     append_variables_using_state_transitions_table function of JointProbabilityMatrix.
#     :rtype : int
#     :param values: list of values which should be in integer range [0, modulo)
#     :param modulo: value self.numvalues usually, e.g. 2 for binary values
#     :return: for binary variables it is the XOR, for others it is summed modulo of integers.
#     """
#     return int(np.mod(np.sum(values), modulo))


### START of testing functions which try to determine if the synergy variables implementation behaves as expected,
### and for which I would like to see the results e.g. to plot in the paper


def test_susceptibilities(num_vars_X, num_vars_Y, num_values, num_samples=50, synergistic=True):

    resp = TestSynergyInRandomPdfs()

    resp.susceptibility_new_local_list = []
    resp.susceptibility_new_global_list = []

    time_before = time.time()

    for sample in xrange(num_samples):
        if not synergistic:
            pdf = JointProbabilityMatrix(num_vars_X + num_vars_Y, num_values)
        else:
            pdf = JointProbabilityMatrix(num_vars_X, num_values)
            pdf.append_synergistic_variables(num_vars_Y)

        num_X1 = int(num_vars_X / 2)

        print 'note: computing old susceptibilities... t=' + str(time.time() - time_before)

        resp.susceptibilities_local_list.append(pdf.susceptibilities_local(num_vars_Y))
        resp.susceptibility_global_list.append(pdf.susceptibility_non_local(range(num_vars_X, len(pdf)), range(num_X1),
                                                                            range(num_X1, num_vars_X)))

        print 'note: computing new susceptibilities... t=' + str(time.time() - time_before)

        resp.susceptibility_new_local_list.append(pdf.susceptibility(range(num_vars_X, len(pdf)), only_non_local=False))
        resp.susceptibility_new_global_list.append(pdf.susceptibility(range(num_vars_X, len(pdf)), only_non_local=True))

        resp.pdf_XY_list.append(pdf.copy())
        resp.total_mi_list.append(pdf.mutual_information(range(num_vars_X), range(num_vars_X, len(pdf))))
        indiv_mis = [pdf.mutual_information([xi], range(num_vars_X, len(pdf))) for xi in range(num_vars_X)]
        resp.indiv_mi_list_list.append(indiv_mis)

        print 'note: finished loop sample=' + str(sample+1) + ' (of ' + str(num_samples) + ') t=' \
              + str(time.time() - time_before)
        print 'note: susceptibilities:', resp.susceptibilities_local_list[-1], ', ', resp.susceptibility_global_list, \
            ', ', resp.susceptibility_new_local_list[-1], ', ', resp.susceptibility_new_global_list[-1]

    return resp


# returned by test_upper_bound_single_srv_entropy()
class TestUpperBoundSingleSRVEntropyResult(object):

    def __init__(self):
        self.num_subject_variables = None
        self.num_synergistic_variables = None
        self.num_values = None
        self.theoretical_upper_bounds = []
        self.entropies_srv = []  # actually lower bounds (same as entropies_lowerbound_srv), namely I(X:SRV) - sum_i I(X_i:SRV)
        self.pdfs_with_srv = []
        self.rel_errors_srv = []  # sum_indiv_mis / total_mi
        self.entropies_lowerbound_srv = []  # I(X:SRV) - sum_i I(X_i:SRV)
        self.entropies_upperbound_srv = []  # I(X:SRV) - max_i I(X_i:SRV)


    # parameters used to produce the results
    num_subject_variables = None
    num_synergistic_variables = None
    num_values = None

    # list of floats, each i'th number should ideally be >= to the i'th element of entropies_srv. These are the
    # theoretically predicted upper bounds of any SRV from the section "Consequential properties" in the synergy
    # paper
    theoretical_upper_bounds = []

    # list of floats, each i'th number is the estimated maximum entropy of a single SRV
    entropies_srv = []  # actually lower bounds (same as entropies_lowerbound_srv), namely I(X:SRV) - sum_i I(X_i:SRV)
    pdfs_with_srv = []
    rel_errors_srv = []  # sum_indiv_mis / total_mi

    entropies_lowerbound_srv = []  # I(X:SRV) - sum_i I(X_i:SRV)
    entropies_upperbound_srv = []  # I(X:SRV) - max_i I(X_i:SRV)

    def __str__(self):
        if len(self.entropies_srv) > 0:
            return '[H(S(x))=' + str(np.mean(self.entropies_srv)) \
                   + '=' + str(np.mean(self.entropies_srv)/np.mean(self.theoretical_upper_bounds)) + 'H_syn(X), +/- ' \
                   + str(np.mean(self.rel_errors_srv)*100.0) + '%]'
        else:
            return '[H(S(x))=nan' \
                   + '=(nan)H_syn(X), +/- ' \
                   + 'nan' + '%]'


def test_upper_bound_single_srv_entropy(num_subject_variables=2, num_synergistic_variables=2, num_values=2,
                                        num_samples=10, tol_rel_err=0.05, verbose=True, num_repeats_per_sample=5):
    """
    Measure the entropy a single SRV and compare it to the theoretical upper bound derived in the synergy paper, along
    with the relative error of the synergy estimation.

    Note: instead of the entropy of the SRV I use the MI of the SRV with the subject variables, because
    the optimization procedure in append_synergistic_variables does not (yet) try to simultaneously minimize
    the extraneous entropy in an SRV, which is entropy in an SRV which does not correlate with any other
    (subject) variable.

    :param num_subject_variables: number of X_i variables to compute an SRV for
    :param num_synergistic_variables:
    :param num_values:
    :param tol_rel_err:
    :param num_samples: number of entropy values will be returned, one for each randomly generated pdf.
    :param verbose:
    :param num_repeats_per_sample: number of optimizations to perform for each sample, trying to get the best result.
    :return: object with a list of estimated SRV entropies and a list of theoretical upper bounds of this same entropy,
    each one for a different, randomly generated PDF.
    :rtype: TestUpperBoundSingleSRVEntropyResult
    """

    result = TestUpperBoundSingleSRVEntropyResult()  # object to hold results

    # parameters used to produce the results
    result.num_subject_variables = num_subject_variables
    result.num_synergistic_variables = num_synergistic_variables
    result.num_values = num_values

    # shorthands
    synergistic_variables = range(num_subject_variables, num_subject_variables + num_synergistic_variables)
    subject_variables = range(num_subject_variables)

    time_before = time.time()

    # generate samples
    for trial in xrange(num_samples):
        pdf = JointProbabilityMatrix(num_subject_variables, numvalues=num_values)

        theoretical_upper_bound = pdf.entropy() - max([pdf.entropy([si]) for si in subject_variables])

        pdf.append_synergistic_variables(num_synergistic_variables, num_repeats=num_repeats_per_sample,
                                         verbose=verbose)

        # prevent double computations:
        indiv_mis = [pdf.mutual_information(synergistic_variables, [si]) for si in subject_variables]
        sum_indiv_mis = sum(indiv_mis)
        total_mi = pdf.mutual_information(synergistic_variables, subject_variables)

        # an error measure of the optimization procedure of finding synergistic variables, potentially it can be quite
        # bad, this one is normalized in [0,1]
        rel_error_srv = (sum_indiv_mis - max(indiv_mis)) / total_mi

        if rel_error_srv <= tol_rel_err:
            # note: instead of the entropy o the SRV I use the MI of the SRV with the subject variables, because
            # the optimization procedure in append_synergistic_variables does not (yet) try to simultaneously minimize
            # the extraneous entropy in an SRV, which is entropy in an SRV which does not correlate with any other
            # (subject) variable.
            # note: subtracting the sum_indiv_mi is done because this part of the entropy of the SRV is not
            # synergistic, so it would overestimate the synergistic entropy. But it is also an estimate, only valid
            # in case the non-synergistic MIs I_indiv(syns : subs) indeed factorizes into the sum, occurring e.g.
            # if the individual subject variables are independent, or if the MI with each subject variable is
            # non-redundant with the MIs with other subject variables.
            # entropy_srv = pdf.entropy(synergistic_variables)

            result.entropies_lowerbound_srv.append(total_mi - sum_indiv_mis)
            result.entropies_upperbound_srv.append(total_mi - max(indiv_mis))
            # take the best estimate for synergistic entropy to be the middle-way between lb and ub
            entropy_srv = (result.entropies_upperbound_srv[-1] + result.entropies_lowerbound_srv[-1]) / 2.0
            result.theoretical_upper_bounds.append(theoretical_upper_bound)
            result.entropies_srv.append(entropy_srv)
            result.pdfs_with_srv.append(pdf)
            result.rel_errors_srv.append(rel_error_srv)

            if verbose:
                print 'note: added a new sample, entropy_srv =', entropy_srv, ' and theoretical_upper_bound =', \
                    theoretical_upper_bound, ', and rel_error_srv =', rel_error_srv
        else:
            if verbose:
                print 'note: trial #' + str(trial) + ' will be discarded because the relative error of the SRV found' \
                      + ' is ' + str(rel_error_srv) + ' which exceeds ' + str(tol_rel_err) \
                      + ' (' + str(time.time() - time_before) + 's elapsed)'

    return result


class TestUpperBoundManySRVEntropyResult(object):

    # three-level dictionary, num_subject_variables_list --> num_synergistic_variables_list --> num_values_list,
    # and values are TestUpperBoundSingleSRVEntropyResult objects
    single_res_dic = dict()


    def plot_rel_err_and_frac_success(self, num_samples=1):

        figs = []

        for num_sub in self.single_res_dic.iterkeys():
            for num_syn in self.single_res_dic[num_sub].iterkeys():
                fig = plt.figure()

                num_vals_list = self.single_res_dic[num_sub][num_syn].keys()
                num_successes = [len(self.single_res_dic[num_sub][num_syn][n].rel_errors_srv) for n in num_vals_list]

                plt.boxplot([self.single_res_dic[num_sub][num_syn][n].rel_errors_srv for n in num_vals_list],
                            labels=map(str, num_vals_list))
                # plt.ylim([0,0.15])
                # plt.xlim([0.5, 4.5])
                plt.xlabel('Number of values per variable')
                plt.ylabel('Relative error of H(S)')

                ax2 = plt.twinx()

                ax2.set_ylabel('Fraction of successful SRVs')
                ax2.plot((1,2,3,4), [s / num_samples for s in num_successes], '-o')
                ax2.set_ylim([0.80, 1.02])

                plt.show()

                figs.append(fig)

        return figs


    def plot_efficiency(self):
        """

        Warning: for num_subject_variables > 2 and num_synergistic_variables < num_subject_variables - 1 then
        the synergistic entropy does not fit in the SRV anyway, so efficiency of 1.0 would be impossible.
        :return: list of figure handles
        """
        figs = []

        for num_sub in self.single_res_dic.iterkeys():
            for num_syn in self.single_res_dic[num_sub].iterkeys():
                fig = plt.figure()

                num_vals_list = self.single_res_dic[num_sub][num_syn].keys()

                # list of lists
                effs = [np.divide(self.single_res_dic[num_sub][num_syn][n].entropies_srv,
                                  self.single_res_dic[num_sub][num_syn][n].theoretical_upper_bounds) for n in (2,3,4,5)]

                plt.boxplot(effs, labels=num_vals_list)
                plt.ylabel('H(S) / H_syn(X)')
                plt.xlabel('Number of values per variable')
                plt.show()

                figs.append(fig)

        return figs


def test_upper_bound_single_srv_entropy_many(num_subject_variables_list=list([2,3]),
                                             num_synergistic_variables_list=list([1,2]),
                                             num_values_list=list([2,3]),
                                             num_samples=10, tol_rel_err=0.05, verbose=True, num_repeats_per_sample=5):

    results_dict = dict()

    time_before = time.time()

    total_num_its = len(num_subject_variables_list) * len(num_synergistic_variables_list) * len(num_values_list)
    num_finished_its = 0

    for num_subject_variables in num_subject_variables_list:
        results_dict[num_subject_variables] = dict()

        for num_synergistic_variables in num_synergistic_variables_list:
            results_dict[num_subject_variables][num_synergistic_variables] = dict()

            for num_values in num_values_list:
                resobj = test_upper_bound_single_srv_entropy(num_subject_variables=num_subject_variables,
                                                             num_synergistic_variables=num_synergistic_variables,
                                                             num_values=num_values,
                                                             num_samples=num_samples,
                                                             tol_rel_err=tol_rel_err,
                                                             verbose=int(verbose)-1,
                                                             num_repeats_per_sample=num_repeats_per_sample)

                results_dict[num_subject_variables][num_synergistic_variables][num_values] = copy.deepcopy(resobj)

                num_finished_its += 1

                if verbose:
                    print 'note: finished', num_finished_its, '/', total_num_its, 'iterations after', \
                        time.time() - time_before, 'seconds. Result for (nx,ns,nv)=' \
                        + str((num_subject_variables, num_synergistic_variables, num_values)) + ': ', resobj

                del resobj  # looking for some inadvertent use of pointer

    return results_dict


class TestSynergyInRandomPdfs(object):

    def __init__(self):
        self.syn_info_list = []
        self.total_mi_list = []
        self.indiv_mi_list_list = []  # list of list
        self.susceptibility_global_list = []
        self.susceptibilities_local_list = []  # list of list
        self.pdf_XY_list = []

    syn_info_list = []
    total_mi_list = []
    indiv_mi_list_list = []  # list of list
    susceptibility_global_list = []
    susceptibilities_local_list = []  # list of list
    pdf_XY_list = []


def test_synergy_in_random_pdfs(num_variables_X, num_variables_Y, num_values,
                                num_samples=10, tolerance_nonsyn_mi=0.05, verbose=True, minimize_method=None,
                                perturbation_size=0.1, num_repeats_per_srv_append=3):

    """


    :param minimize_method: the default is chosen if None, which is good, but not necessarily fast. One other, faster
    option I found was "SLSQP", but I have not quantified rigorously how much better/worse in terms of accuracy it is.
    :param num_variables_X:
    :param num_variables_Y:
    :param num_values:
    :param num_samples:
    :param tolerance_nonsyn_mi:
    :param verbose:
    :rtype: TestSynergyInRandomPdfs
    """

    result = TestSynergyInRandomPdfs()

    time_before = time.time()

    try:
        for i in xrange(num_samples):
            pdf = JointProbabilityMatrix(num_variables_X + num_variables_Y, num_values)

            result.pdf_XY_list.append(pdf)

            result.syn_info_list.append(pdf.synergistic_information(range(num_variables_X, num_variables_X + num_variables_Y),
                                                                    range(num_variables_X),
                                                                    tol_nonsyn_mi_frac=tolerance_nonsyn_mi,
                                                                    verbose=bool(int(verbose)-1),
                                                                    minimize_method=minimize_method,
                                                                    num_repeats_per_srv_append=num_repeats_per_srv_append))

            result.total_mi_list.append(pdf.mutual_information(range(num_variables_X, num_variables_X + num_variables_Y),
                                                          range(num_variables_X)))

            indiv_mi_list = [pdf.mutual_information(range(num_variables_X, num_variables_X + num_variables_Y),
                                                          [xid]) for xid in xrange(num_variables_X)]

            result.indiv_mi_list_list.append(indiv_mi_list)

            result.susceptibility_global_list.append(pdf.susceptibility_global(num_variables_Y, ntrials=50,
                                                                               perturbation_size=perturbation_size))

            result.susceptibilities_local_list.append(pdf.susceptibilities_local(num_variables_Y, ntrials=50,
                                                                                 perturbation_size=perturbation_size))

            if verbose:
                print 'note: finished sample', i, 'of', num_samples, ', syn. info. =', result.syn_info_list[-1], 'after', \
                    time.time() - time_before, 'seconds'
    except KeyboardInterrupt as e:
        min_len = len(result.susceptibilities_local_list)  # last thing to append to in above loop, so min. length

        result.syn_info_list = result.syn_info_list[:min_len]
        result.total_mi_list = result.total_mi_list[:min_len]
        result.indiv_mi_list_list = result.indiv_mi_list_list[:min_len]
        result.susceptibility_global_list = result.susceptibility_global_list[:min_len]

        if verbose:
            print 'note: keyboard interrupt. Will stop the loop and return the', min_len, 'result I have so far.', \
                'Took me', time.time() - time_before, 'seconds.'

    return result


# returned by test_accuracy_orthogonalization()
class TestAccuracyOrthogonalizationResult(object):
    # parameters used to produce the results
    num_subject_variables = None
    num_orthogonalized_variables = None
    num_values = None

    # list of floats, each i'th number is the estimated entropy of the <num_subject_variables> variables
    actual_entropies_subject_variables = []
    actual_entropies_orthogonalized_variables = []
    actual_entropies_parallel_variables = []
    actual_entropies_orthogonal_variables = []

    # list of JointProbabilityMatrix objects
    joint_pdfs = []
    # list of floats, each i'th number is the relative error of the orthogonalization, in range 0..1
    rel_errors = []


def test_accuracy_orthogonalization(num_subject_variables=2, num_orthogonalized_variables=2, num_values=2,
                                        num_samples=10, verbose=True, num_repeats=5):
    """


    Note: instead of the entropy o the SRV I use the MI of the SRV with the subject variables, because
    the optimization procedure in append_synergistic_variables does not (yet) try to simultaneously minimize
    the extraneous entropy in an SRV, which is entropy in an SRV which does not correlate with any other
    (subject) variable.

    :param num_subject_variables:
    :param num_synergistic_variables:
    :param num_values:
    :param tol_rel_err:
    :return: object with a list of estimated SRV entropies and a list of theoretical upper bounds of this same entropy,
    each one for a different, randomly generated PDF.
    :rtype: TestUpperBoundSingleSRVEntropyResult
    """

    # todo: also include the option of known null hypothesis, to see what is the error if it is already known that 0.0
    # error is in fact possible

    # take these to be equal to num_orthogonalized_variables just to be sure
    num_orthogonal_variables = num_orthogonalized_variables
    num_parallel_variables = num_orthogonalized_variables

    result = TestAccuracyOrthogonalizationResult()  # object to hold results

    # parameters used to produce the results
    result.num_subject_variables = num_subject_variables
    result.num_synergistic_variables = num_orthogonalized_variables
    result.num_values = num_values

    # shorthands for variable index ranges for the different roles of variables
    orthogonalized_variables = range(num_subject_variables, num_subject_variables + num_orthogonalized_variables)
    subject_variables = range(num_subject_variables)
    orthogonal_variables = range(num_subject_variables + num_orthogonalized_variables,
                                 num_subject_variables + num_orthogonalized_variables + num_orthogonal_variables)
    parallel_variables = range(num_subject_variables + num_orthogonalized_variables + num_orthogonal_variables,
                               num_subject_variables + num_orthogonalized_variables + num_orthogonal_variables
                               + num_parallel_variables)

    time_before = time.time()

    # generate samples
    for trial in xrange(num_samples):
        pdf = JointProbabilityMatrix(num_subject_variables + num_orthogonalized_variables, numvalues=num_values)

        pdf.append_orthogonalized_variables(orthogonalized_variables, num_orthogonal_variables, num_parallel_variables,
                                            verbose=verbose, num_repeats=num_repeats)

        assert len(pdf) == num_subject_variables + num_orthogonalized_variables + num_orthogonal_variables \
                           + num_parallel_variables

        result.joint_pdfs.append(pdf)

        result.actual_entropies_subject_variables.append(pdf.entropy(subject_variables))
        result.actual_entropies_orthogonalized_variables.append(pdf.entropy(orthogonalized_variables))
        result.actual_entropies_orthogonal_variables.append(pdf.entropy(orthogonal_variables))
        result.actual_entropies_parallel_variables.append(pdf.entropy(parallel_variables))

        rel_error_1 = (pdf.mutual_information(orthogonalized_variables, subject_variables) - 0) \
                      / result.actual_entropies_subject_variables[-1]
        rel_error_2 = (pdf.mutual_information(orthogonalized_variables, subject_variables)
                      - pdf.mutual_information(parallel_variables, subject_variables)) \
                      / pdf.mutual_information(orthogonalized_variables, subject_variables)
        rel_error_3 = (result.actual_entropies_orthogonalized_variables[-1]
                      - pdf.mutual_information(orthogonal_variables + parallel_variables, orthogonalized_variables)) \
                      / result.actual_entropies_orthogonalized_variables[-1]

        # note: each relative error term is intended to be in range [0, 1]

        rel_error = rel_error_1 + rel_error_2 + rel_error_3

        # max_rel_error = result.actual_entropies_subject_variables[-1] \
        #                 + pdf.mutual_information(orthogonalized_variables, subject_variables) \
        #                 + result.actual_entropies_orthogonalized_variables[-1]
        max_rel_error = 3.0

        rel_error /= max_rel_error

        result.rel_errors.append(rel_error)

        print 'note: finished trial #' + str(trial) + ' after', time.time() - time_before, 'seconds. rel_error=', \
            rel_error, '(' + str((rel_error_1, rel_error_2, rel_error_3)) + ')'

    return result


# todo: make a test function which iteratively appends a set of SRVs until no more can be found (some cut-off MI),
# then I am interested in the statistics of the number of SRVs that are found, as well as the statistics of whether
# the SRVs are statistically independent or not, and whether they have synergy about each other (if agnostic_about
# passed to append_syn* is a list of lists) and whether indeed every two SRVs necessarily will have info about an
# individual Y_i. In the end, if the assumption of perfect orthogonal decomposition of any RVs too strict, or is
# is often not needed at all? Because it would only be needed in case there are at least two SRVs, the combination of
# which (combined RV) has more synergy about Y than the two individually summed, but also the combined RV has
# info about individual Y_i so it is not an SRV by itself, so the synergy contains both synergistic and individual info.

def test_num_srvs(num_subject_variables=2, num_synergistic_variables=2, num_values=2, num_samples=10,
                  verbose=True, num_repeats=5):
    assert False


def test_impact_perturbation(num_variables_X=2, num_variables_Y=1, num_values=5,
                                num_samples=20, tolerance_nonsyn_mi=0.05, verbose=True, minimize_method=None,
                                perturbation_size=0.1, num_repeats_per_srv_append=3):

    """


    :param minimize_method: the default is chosen if None, which is good, but not necessarily fast. One other, faster
    option I found was "SLSQP", but I have not quantified rigorously how much better/worse in terms of accuracy it is.
    :param num_variables_X:
    :param num_variables_Y:
    :param num_values:
    :param num_samples:
    :param tolerance_nonsyn_mi:
    :param verbose:
    :rtype: TestSynergyInRandomPdfs
    """

    result = TestSynergyInRandomPdfs()

    time_before = time.time()

    try:
        for i in xrange(num_samples):
            pdf = JointProbabilityMatrix(num_variables_X, num_values)
            pdf.append_synergistic_variables(num_variables_Y, num_repeats=num_repeats_per_srv_append)

            pdf_Y_cond_X = pdf.conditional_probability_distributions(range(len(num_variables_X)))

            result.pdf_XY_list.append(pdf)

            result.syn_info_list.append(pdf.synergistic_information(range(num_variables_X, num_variables_X + num_variables_Y),
                                                                    range(num_variables_X),
                                                                    tol_nonsyn_mi_frac=tolerance_nonsyn_mi,
                                                                    verbose=bool(int(verbose)-1),
                                                                    minimize_method=minimize_method,
                                                                    num_repeats_per_srv_append=num_repeats_per_srv_append))

            result.total_mi_list.append(pdf.mutual_information(range(num_variables_X, num_variables_X + num_variables_Y),
                                                          range(num_variables_X)))

            indiv_mi_list = [pdf.mutual_information(range(num_variables_X, num_variables_X + num_variables_Y),
                                                          [xid]) for xid in xrange(num_variables_X)]

            result.indiv_mi_list_list.append(indiv_mi_list)

            result.susceptibility_global_list.append(pdf.susceptibility_global(num_variables_Y, ntrials=50,
                                                                               perturbation_size=perturbation_size))

            result.susceptibilities_local_list.append(pdf.susceptibilities_local(num_variables_Y, ntrials=50,
                                                                                 perturbation_size=perturbation_size))

            if verbose:
                print 'note: finished sample', i, 'of', num_samples, ', syn. info. =', result.syn_info_list[-1], 'after', \
                    time.time() - time_before, 'seconds'
    except KeyboardInterrupt as e:
        min_len = len(result.susceptibilities_local_list)  # last thing to append to in above loop, so min. length

        result.syn_info_list = result.syn_info_list[:min_len]
        result.total_mi_list = result.total_mi_list[:min_len]
        result.indiv_mi_list_list = result.indiv_mi_list_list[:min_len]
        result.susceptibility_global_list = result.susceptibility_global_list[:min_len]

        if verbose:
            print 'note: keyboard interrupt. Will stop the loop and return the', min_len, 'result I have so far.', \
                'Took me', time.time() - time_before, 'seconds.'

    return result
