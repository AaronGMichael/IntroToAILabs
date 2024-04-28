from pgmpy.models import BayesianNetwork as BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

bayesNet = BayesianModel()

bayesNet.add_nodes_from(['L', 'N', 'I', 'S', 'R'])
bayesNet.add_edges_from([('L', 'N'), ('I', 'N'), ('S', 'N'), ('N', 'R'), ('S', 'R')])

I_cpd = TabularCPD('I', 2, values=[[0.21], [0.79]],
                   state_names={'I': ['Inappropriate/Plagarism', 'Valid']})
L_cpd = TabularCPD('L', 2, values=[[0.08], [0.92]],
                   state_names={'L': ['Signalled', 'Not Signalled']})
S_cpd = TabularCPD('S', 2, values=[[0.12], [0.88]],
                   state_names={'S': ['Suspended', 'Valid']})

N_cpd = TabularCPD('N', 2,
                   values=[[0.92, 0.88, 0.79, 0.73, 0.22, 0.08, 0.17, 0.03],
                           [0.08, 0.12, 0.21, 0.27, 0.78,0.92, 0.83, 0.97]],
                   evidence=['L', 'S', 'I'], evidence_card=[2, 2, 2],
                   state_names={
                       'N': ['Not Scored', 'Scored'], 'L': ['Signalled', 'Not Signalled'],
                       'S': ['Suspended', 'Valid'], 'I': ['Inappropriate/Plagarism', 'Valid']
                   })

R_cpd = TabularCPD('R', 2, values=[[0.38, 0.08, 0.08, 0.05], [0.62, 0.92, 0.92, 0.95]],
                   evidence=['N', 'S'], evidence_card=[2, 2],
                   state_names={'R': ['Rattrapage', 'No'],
                                'N': ['Not Scored', 'Scored'],
                                'S': ['Suspended', 'Valid']}
                   )


bayesNet.add_cpds(I_cpd, L_cpd, S_cpd, N_cpd, R_cpd)

print(bayesNet.check_model())

solver = VariableElimination(bayesNet)
print(bayesNet.get_independencies())
# print(solver.query(['N'], evidence= {'L' : "Signalled"}))
# print(solver.query(['N']))

