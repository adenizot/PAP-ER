# PAP-ER
This is the repository that contains the code associated with the paper below:

Denizot, A.; Veloz Castillo, M. F.; Puchenkov, P.; Cali, C.; De Schutter, E. (2025). The ultrastructural properties of the endoplasmic reticulum govern microdomain signaling in perisynaptic astrocytic processes. DOI: 10.1101/2022.02.28.482292

Files are organized within different folders, as follows:
- 'Ca2+Model' contains the simulation code used in Fig. 3-5 of the paper. The code was written for simulations using the STochastic Engine for Pathway Simulations (STEPS, http://steps.sourceforge.net/STEPS/default.php) 3.5.0.
- 'GeometryAnalysis' contains the analysis code used to measure PM-PSD, ER-PSD or ER-PM distances.
- 'PAPMeshGeneration' contains the algorithm that allows the generation of realistic PAP meshes with various ER distributions in the PAP, with constant ER and PAP shape.

The codes in 'GeometryAnalysis' and 'PAPMeshGeneration' were implemented for Blender 4.3.2. 

Datasets are shared on Zenodo under the Creative Commons Attribution 4.0 International license: https://zenodo.org/records/17106549. They are organized into two main folders:
- 'PAPMeshes', which contains the perisynaptic astrocytic meshes used in this study
- 'TripartiteSynapseMeshes', which contains the tripartite synapse meshes used in this study

Contact:
audrey.denizot@inria.fr
