import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import steps.geom as stetmesh
import steps.rng as srng
import steps.solver as ssolver
import camodel
from random import randrange as randr

# Set simulation parameters
NITER = 1
T_END = 100.0
DT = 0.001
POINTS = int(T_END / DT)
tpnts = np.arange(0.0, T_END, DT)
ntpnts = tpnts.shape[0]

# Create random number generator
seed = int(sys.argv[1])
r = srng.create('mt19937', 512)
r.initialize(seed)

# Get PAP mesh name
mesh = str(sys.argv[2])

# Import the Ca2+ activity model
mdl = camodel.getModel()

# Import the 3D geometry
mesh, cyt_tris, er_tris, er_patch, cyto_patch, er_area, cyto_area = camodel.gen_geom(mesh)

# Create a solver object
sim = ssolver.Tetexact(mdl, mesh, r)

# Run the simulation
# Reset the simulation object
sim.reset()

# Set initial conditions
nb_ip3r = 230 # value depends on ER surface area as IP3R density is constant: 3.5e-3 per um2
sim.setCompConc('cyto', 'ca', 120e-9)
sim.setCompConc('cyto', 'ip3', 120e-9)
nb_plc = int(cyto_area * 1696 / 6.88566421434e-13)
sim.setPatchCount('cyto_patch', 'plc', nb_plc)

# Optional: uncomment to add Ca2+ indicators to the cytosolic model to measure Ca-GCaMP concentration variations
# Note that steady state is reached after several seconds of simulation time and should be reached before performing analysis of Ca2+ activity
#sim.setCompConc('cyto', 'GCaMP6s', 9.55e-6)
#sim.setCompConc('cyto', 'ca_GCaMP6s', 450e-9)

# Number of clusters of IP3Rs on the ER
# Important: should be a divider of nb_ip3r
nb_clust = int(sys.argv[3])

# Number of IP3 molecules infused at stimulation time
ip3_infused = int(sys.argv[4])

# Whether Ca2+ channels at the plasma membrane are co-clustered with IP3R or not
cocl = int(sys.argv[5])

###################################
#### IP3R CLUSTERING ON THE ER ####
###################################
# In this script, IP3R clusters are positioned randomly on the surface of the ER
nb_ip3r_per_clust = nb_ip3r / nb_clust
clusters_centers = []
pump_tris = cyt_tris
IP3R_tris = er_tris
clusters_centers = []

# Determine the location of each IP3R cluster on the ER membrane
for i in range(0, nb_clust):
    list_tris_ROI = []
    center_found = False
    tri_index = randr(0, len(IP3R_tris) - 1)
    # tri_center = er_tri_idx[tri_index]
    tri_center = IP3R_tris[tri_index]
    if i >= 1:
        while center_found == False:
            center_found = True
            center_baryc = mesh.getTriBarycenter(tri_center)
            for j in range(0, len(clusters_centers) - 1):
                tri_baryc = mesh.getTriBarycenter(clusters_centers[j])
                dist = math.sqrt(math.pow((center_baryc[0] - tri_baryc[0]), 2) \
                                 + math.pow((center_baryc[1] - tri_baryc[1]), 2) \
                                 + math.pow((center_baryc[2] - tri_baryc[2]), 2))

                # If the triangle selected is < 30 nm to the center of other IP3R cluster ROIs, search for a new location
                if dist < 3e-8:
                    center_found = False
                    tri_index = randr(0, len(IP3R_tris) - 1)
                    tri_center = IP3R_tris[tri_index]
                # Otherwise, select the triangle as the center of the new IP3R cluster ROI
                else:
                    center_found = True

    clusters_centers.append(tri_center)
    list_tris_ROI.append(tri_center)
    # Remove the center from the pool of triangles on which future IP3R clusters can be located
    IP3R_tris.remove(tri_center)

    # keep only triangles that are in IP3R_tris = not already in a cluster + on ER membrane and not on top circles
    for neighb in mesh.getTriTriNeighbs(tri_center):
        if neighb in IP3R_tris:
            list_tris_ROI.append(neighb)
            # Remove the ROI triangles from the pool of triangles on which future IP3R clusters can be located
            IP3R_tris.remove(neighb)
    # Create regions of interest (ROIs), as defined in STEPS, for each IP3R cluster
    mesh.addROI(str(i), stetmesh.ELEM_TRI, list_tris_ROI)
    sim.setROICount(str(i), 'unb_IP3R', nb_ip3r_per_clust)

##############################################
###### CA SOURCES CO-CLUSTERING SCRIPT #######
##############################################
# No co-clustering: random distribution of Ca2+ channels on the plasma membrane
if cocl == 0:
    sim.setPatchCount('cyto_patch', 'ca_ch', nb_ip3r)
else:
    # Co-clustering of Ca2+ channels: locate Ca2+ channels on the closest triangles of the plasma membrane (PM) to IP3R cluster sites
    plasmamb_tri_idx = cyto_patch.getAllTriIndices()
    pump_tris = cyt_tris

    # Create the same number of clusters of Ca2+ channels on the PM as IP3R clusters on the ER membrane
    for j in range(0, nb_clust):
        list_mb_tris_ROI = []
        # Get ROI triangles of the IP3R cluster
        er_tris = mesh.getROITris(str(j))
        # Get the center of the IP3R cluster ROI
        er_roi_center = clusters_centers[j]
        er_roi_cent_baryc = mesh.getTriBarycenter(er_roi_center)
        # Get the closest plasma membrane triangle to the center of the IP3R cluster ROI center
        mb_tri_dist_table = []
        for tr in pump_tris:
            tr_baryc = mesh.getTriBarycenter(tr)
            # Compute radial distance the PM triangle to the IP3R cluster ROI center
            tr_IP3R_dist = math.sqrt(math.pow((er_roi_cent_baryc[0] - tr_baryc[0]), 2) \
                                     + math.pow((er_roi_cent_baryc[1] - tr_baryc[1]), 2) \
                                     + math.pow((er_roi_cent_baryc[2] - tr_baryc[2]), 2))
            mb_tri_dist_table.append(tr_IP3R_dist)

        # Create a ROI containing the plasma membrane triangles that form the Ca2+ channel cluster
        tr_arg = np.argmin(mb_tri_dist_table)
        center_mb_cluster = pump_tris[tr_arg]
        pump_tris.remove(center_mb_cluster)
        list_mb_tris_ROI.append(center_mb_cluster)
        for neighb in mesh.getTriTriNeighbs(center_mb_cluster):
            if neighb in pump_tris:
                list_mb_tris_ROI.append(neighb)
                pump_tris.remove(neighb)
        name = 'mb_clust_' + str(j)
        mesh.addROI(name, stetmesh.ELEM_TRI, list_mb_tris_ROI)
        sim.setROICount(name, 'ca_ch', nb_ip3r_per_clust)

###################################
##### IP3 INFUSION SITE ###########
###################################
# Script that defines the tetrahedra where IP3 will be infused at stimulation time
min_xcoord = mesh.getBoundMin()[0]
max_xcoord = mesh.getBoundMax()[0]
len_geom = max_xcoord - min_xcoord

stris = cyt_tris
submemb_tets = []
for tri in stris:
    neighbs = mesh.getTriTetNeighb(tri)
    if neighbs[0] == -1:
        submemb_tets.append(neighbs[1])
    else:
        submemb_tets.append(neighbs[0])
contact_tets = []
for tet in submemb_tets:
    coords = mesh.getTetBarycenter(tet)
    # if coords[2] > 4e-7:
    if coords[0] < min_xcoord + 0.1 * len_geom:
        contact_tets.append(tet)

# Create the ROI where IP3 will be infused at stimulation time
mesh.addROI('process_tip', stetmesh.ELEM_TET, contact_tets)

# Create files in which Ca2+, IP3R and IP3 dynamics will be recorded at each time step
f = open("freeca_d9s4a34b1_cl%d_ip%d_cocl%d.%d" % (nb_clust, ip3_infused, cocl, seed), "w")
fip = open("nbopip3rpercl_d9s4a34b1_cl%d_ip%d_cocl%d.%d" % (nb_clust, ip3_infused, cocl, seed), "w")

# Run the simulation and record relevant data
for i in range(ntpnts):
    sim.run(tpnts[i])
    if i == 1000:
        # Inject IP3 in the 'process_tip' ROI, at time t=1s
        sim.setROICount('process_tip', 'ip3', ip3_infused)
    # Record Ca2+ and IP3 concentration in the PAP cytosol as well as the number of open IP3R at the membrane of the ER at each time step    
    f.write("%d %d %d\n" % (sim.getCompCount('cyto', 'ca'), sim.getPatchCount('er_patch', 'open_IP3R'),
                            sim.getCompCount('cyto', 'ip3')))
    # Record the number of IP3R channels in the open state within each cluster at each time step                              
    for i in range(0, nb_clust):
        ROI_surf_name = str(i)
        nb_ip3r_ROI_i = sim.getROICount(ROI_surf_name, 'open_IP3R')
        fip.write('%d ' % nb_ip3r_ROI_i)
    fip.write('\n')
    f.flush()
    fip.flush()

f.close()
fip.close()

