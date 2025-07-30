# Networks with Synergistic Interactions

This GitHub repository contains the code for the paper "Understanding multimorbidity requires sign-disease networks and higher-order interactions, a perspective. " by Cillian Hourican, Geeske Peeters, René Melis, Thomas M. Gill, Marcel Olde Rikkert, Rick Quax.
[[Paper Here]](https://doi.org/10.3389/fsysb.2023.1155599)

The data folder contains data generated from the synthetic model presented in the paper. The Construct_network.py script illustrates how such a model can be recreated, while Oinfo.py illustrates how O-information values can be computed from a dataset. Note this code only works for discrete variables. 

Network construction requires the jointpdf package (https://bitbucket.org/rquax/jointpdf/src/master/).
To ease implementation, we have provided some source files from this package. Please refer to that package for documentation. Some functions in jointpdf/ are reproduced from jointpdf by R. Quax (original license preserved)

To create a synergistic triplet with two independent nodes with uniform distributions, where each variable takes 4 possible states, we could write

```ruby
num_val = 4
a = JointProbabilityMatrix(num_val, num_val, joint_probs="uniform")\n
b = JointProbabilityMatrix(num_val, num_val, joint_probs="uniform")
tf.append_independent_variables(a, b)
tf.append_synergistic_variables(a, 1, subject_variables=[0, 1])

# We can now generate data using
data = a.generate_samples(1000)
data = np.array(data)
pd.DataFrame(data).to_csv("synergistic_interaction_example.csv")
```

Note: A python-based toolbox for effeciently creating networks with synergistic interactions, finding synergistic associations in data using O-information and SRVs, and creating subsequent hypergraphs is currently in progress. Once released, a link wil be provided here. 
