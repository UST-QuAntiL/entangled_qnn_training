# Composition of Entangled Training Data in Quantum Neural Network Training
Experiment/Code for reproduction of results for Composition of Entangled Training Data in Quantum Neural Network Training.

Experiments:
- avg_rank_exp.py: Experiments for training QNNs using training data of varying Schmidt rank
- nlihx_exp.py: Experiments for training QNNs using linearly dependent (in $\mathcal{H}_X$) data
- ortho_exp.py: Experiments for training QNNs using orthogonal training data
	
Visualisation/Analysis of data (``plots.py``):
- Generates plots for above experiments
	+ either from the data in ``experimental_results`` or from the processed results (see Data)
	+ processes results to extract information from raw data in ``experimental_results`` (to change behavior see the function calls at the end of ``plots.py``)
	
The data is available for download at **TODO**



#### Disclaimer of Warranty

Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.
