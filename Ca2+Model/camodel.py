# coding=utf-8
import steps.model as smod
import steps.geom as stetmesh
import steps.utilities.meshio as smeshio
import steps.rng as rng
import steps.utilities.meshio as meshio
from numpy import math
import sys
import numpy as np
import networkx as nx

def diff(list1, list2):
    return list(set(list1).symmetric_difference(set(list2)))


def getModel():
    # Create model container
    mdl = smod.Model()

    # Create chemical species
    ca = smod.Spec('ca', mdl)
    ip3 = smod.Spec('ip3', mdl)
    plc = smod.Spec('plc', mdl)

    # Calcium indicators: GCaMP6s
    GCaMP6s = smod.Spec('GCaMP6s', mdl)
    ca_GCaMP6s = smod.Spec('ca_GCaMP6s', mdl)


    # Create species of all possible IP3R states
    unb_IP3R = smod.Spec('unb_IP3R', mdl)
    ip3_IP3R = smod.Spec('ip3_IP3R', mdl)
    caa_IP3R = smod.Spec('caa_IP3R', mdl)
    cai_IP3R = smod.Spec('cai_IP3R', mdl)
    open_IP3R = smod.Spec('open_IP3R', mdl)
    cai_ip3_IP3R = smod.Spec('cai_ip3_IP3R', mdl)
    ca2_IP3R = smod.Spec('ca2_IP3R', mdl)
    ca2_ip3_IP3R = smod.Spec('ca2_ip3_IP3R', mdl)

    # ER surface system
    ssys = smod.Surfsys('ssys', mdl)

    # Plasma membrane surface system
    mb_surf = smod.Surfsys('mb_surf', mdl)

    # Cytosolic volume system
    vsys = smod.Volsys('vsys', mdl)

    ##### DEFINE DIFFUSION RULES #####
    # Diffusion constants
    # Diffusion constant of Calcium (buffered)
    DCST = 0.013e-9
    # Diffusion constant of IP3
    DIP3 = 0.280e-9
    # Diffusion constant of GCaMP6s
    DGCAMP = 0.050e-9

    diff_freeca = smod.Diff('diff_freeca', vsys, ca, DCST)
    diff_ip3 = smod.Diff('diff_ip3', vsys, ip3, DIP3)

    diff_GCaMP6s = smod.Diff('diff_GCaMP6s', vsys, GCaMP6s, DGCAMP)
    diff_ca_GCaMP6s = smod.Diff('diff_ca_GCaMP6s', vsys, ca_GCaMP6s, DGCAMP)

    # Create Ca2+ channels that will be located on the plasma membrane
    ca_ch = smod.Spec('ca_ch', mdl)


    #####  DEFINE REACTIONS  #####
    ####Calcium Influx and Buffering Reactions######
    # Ca2+ degradation
    ca_deg = smod.Reac('ca_deg', vsys, lhs=[ca])


    # Ca2+ influx via Ca2+ channels at the plasma membrane
    ca_leak = smod.SReac('ca_leak', mb_surf, slhs=[ca_ch], srhs=[ca_ch], irhs=[ca])

    # Ca2+ binding to GCaMP6s indicators
    GCaMP6s_bind_ca_f = smod.Reac('GCaMP6s_bind_ca_f', vsys, \
                                lhs=[ca, GCaMP6s], rhs=[ca_GCaMP6s])
    GCaMP6s_bind_ca_b = smod.Reac('GCaMP6s_bind_ca_b', vsys, \
                                lhs=[ca_GCaMP6s], rhs=[GCaMP6s, ca])

    #### IP3-RELATED REACTIONS ######
    # IP3 degradation
    ip3_deg = smod.Reac('ip3_deg', vsys, lhs=[ip3])

    # Ca2+-dependent IP3 synthesis by PLCdelta
    plc_ip3_synthesis = smod.SReac('plc_ip3_synthesis', mb_surf, \
                                   slhs=[plc], ilhs= [ca], srhs=[plc], irhs= [ca, ip3])

    #### IP3R kinetics #####
    # surface/volume reaction ca from cytosol binds activating IP3R site on unbound IP3R
    unb_IP3R_bind_caa_f = smod.SReac('unb_IP3R_bind_caa_f', ssys,\
                                     ilhs=[ca], slhs=[unb_IP3R], srhs=[caa_IP3R])
    unb_IP3R_bind_caa_b = smod.SReac('unb_IP3R_bind_caa_b', ssys, \
                                     slhs=[caa_IP3R], srhs=[unb_IP3R], irhs=[ca])

    # surface/volume reaction ca from cytosol binds inactivating IP3R site on unbound IP3R
    unb_IP3R_bind_cai_f = smod.SReac('unb_IP3R_bind_cai_f', ssys, \
                                     ilhs=[ca], slhs=[unb_IP3R], srhs=[cai_IP3R])
    unb_IP3R_bind_cai_b = smod.SReac('unb_IP3R_bind_cai_b', ssys, \
                                     slhs=[cai_IP3R], srhs=[unb_IP3R], irhs=[ca])

    # surface/volume reaction ca from cytosol binds activating IP3R site on caa_IP3R
    caa_IP3R_bind_ca_f = smod.SReac('caa_IP3R_bind_ca_f', ssys, \
                                    ilhs=[ca], slhs=[caa_IP3R], srhs=[ca2_IP3R])
    caa_IP3R_bind_ca_b = smod.SReac('caa_IP3R_bind_ca_b', ssys, \
                                    slhs=[ca2_IP3R], srhs=[caa_IP3R], irhs=[ca])

    # surface/volume reaction ca from cytosol binds activating IP3R site on ip3_IP3R
    ip3_IP3R_bind_caa_f = smod.SReac('ip3_IP3R_bind_caa_f', ssys, \
                                     ilhs=[ca], slhs=[ip3_IP3R], srhs=[open_IP3R])
    ip3_IP3R_bind_caa_b = smod.SReac('ip3_IP3R_bind_caa_b', ssys, \
                                     slhs=[open_IP3R], srhs=[ip3_IP3R], irhs=[ca])

    # surface/volume reaction ca from cytosol binds inactivating IP3R site on ip3_IP3R
    ip3_IP3R_bind_cai_f = smod.SReac('ip3_IP3R_bind_cai_f', ssys, \
                                     ilhs=[ca], slhs=[ip3_IP3R], srhs=[cai_ip3_IP3R])
    ip3_IP3R_bind_cai_b = smod.SReac('ip3_IP3R_bind_cai_b', ssys, \
                                     slhs=[cai_ip3_IP3R], srhs=[ip3_IP3R], irhs=[ca])

    # surface/volume reaction ca from cytosol binds activating IP3R site on cai_IP3R
    cai_IP3R_bind_ca_f = smod.SReac('cai_IP3R_bind_ca_f', ssys, \
                                    ilhs=[ca], slhs=[cai_IP3R], srhs=[ca2_IP3R])
    cai_IP3R_bind_ca_b = smod.SReac('cai_IP3R_bind_ca_b', ssys, \
                                    slhs=[ca2_IP3R], srhs=[cai_IP3R], irhs=[ca])

    # surface/volume reaction ca from cytosol binds inactivating IP3R site on open_IP3R
    open_IP3R_bind_ca_f = smod.SReac('open_IP3R_bind_ca_f', ssys, \
                                     ilhs=[ca], slhs=[open_IP3R], srhs=[ca2_ip3_IP3R])
    open_IP3R_bind_ca_b = smod.SReac('open_IP3R_bind_ca_b', ssys, \
                                     slhs=[ca2_ip3_IP3R], srhs=[open_IP3R], irhs=[ca])

    # surface/volume reaction ip3 from cytosol binds unb_IP3R
    unb_IP3R_bind_ip3_f = smod.SReac('unb_IP3R_bind_ip3_f', ssys, \
                                     ilhs=[ip3], slhs=[unb_IP3R], srhs=[ip3_IP3R])
    unb_IP3R_bind_ip3_b = smod.SReac('unb_IP3R_bind_ip3_b', ssys, \
                                     slhs=[ip3_IP3R], srhs=[unb_IP3R], irhs=[ip3])

    # surface/volume reaction ip3 from cytosol binds caa_IP3R
    caa_IP3R_bind_ip3_f = smod.SReac('caa_IP3R_bind_ip3_f', ssys, \
                                     ilhs=[ip3], slhs=[caa_IP3R], srhs=[open_IP3R])
    caa_IP3R_bind_ip3_b = smod.SReac('caa_IP3R_bind_ip3_b', ssys, \
                                     slhs=[open_IP3R], srhs=[caa_IP3R], irhs=[ip3])

    # surface/volume reaction ip3 from cytosol binds cai_IP3R
    cai_IP3R_bind_ip3_f = smod.SReac('cai_IP3R_bind_ip3_f', ssys, \
                                     ilhs=[ip3], slhs=[cai_IP3R], srhs=[cai_ip3_IP3R])
    cai_IP3R_bind_ip3_b = smod.SReac('cai_IP3R_bind_ip3_b', ssys, \
                                     slhs=[cai_ip3_IP3R], srhs=[cai_IP3R], irhs=[ip3])

    cai_ip3_IP3R_bind_ca_f = smod.SReac('cai_ip3_IP3R_bind_ca_f', ssys, \
                                        ilhs=[ca], slhs=[cai_ip3_IP3R], srhs=[ca2_ip3_IP3R])
    cai_ip3_IP3R_bind_ca_b = smod.SReac('cai_ip3_IP3R_bind_ca_b', ssys, \
                                        slhs=[ca2_ip3_IP3R], srhs=[cai_ip3_IP3R], irhs=[ca])

    # surface/volume reaction ip3 from cytosol binds ca2_IP3R
    ca2_IP3R_bind_ip3_f = smod.SReac('ca2_IP3R_bind_ip3_f', ssys, \
                                     ilhs=[ip3], slhs=[ca2_IP3R], srhs=[ca2_ip3_IP3R])
    ca2_IP3R_bind_ip3_b = smod.SReac('ca2_IP3R_bind_ip3_b', ssys, \
                                     slhs=[ca2_ip3_IP3R], srhs=[ca2_IP3R], irhs=[ip3])

    # Ca2+ influx through an open IP3R channel
    Ca_IP3R_flux = smod.SReac('R_Ca_channel_f', ssys, \
                              slhs=[open_IP3R], irhs=[ca], srhs=[open_IP3R])

    ### REACTION CONSTANTS ###
    # Ca2+ binding to GCaMP6s indicators
    GCaMP6s_bind_ca_f.setKcst(7.78e6)
    GCaMP6s_bind_ca_b.setKcst(1.12)

    # Ca2+ degradation
    ca_deg.setKcst(30)

    # Ca2+ leak
    ca_leak.setKcst(3e-2)

    #### IP3 Influx and Buffering Reactions ######
    # IP3 degradation
    ip3_deg.setKcst(1.2e-4)
    
    # Ca2+ dependent IP3 synthesis by PLCdelta
    plc_ip3_synthesis.setKcst(1)

    #### IP3R kinetics #####
    caa_f = 1.2e6
    cai_f = 1.6e4
    ip3_f = 4.1e7
    caa_b = 5e1
    cai_b = 1e2
    ip3_b = 4e2
    unb_IP3R_bind_caa_f.setKcst(caa_f)
    unb_IP3R_bind_caa_b.setKcst(caa_b)

    unb_IP3R_bind_cai_f.setKcst(cai_f)
    unb_IP3R_bind_cai_b.setKcst(cai_b)

    caa_IP3R_bind_ca_f.setKcst(cai_f)
    caa_IP3R_bind_ca_b.setKcst(cai_b)

    ip3_IP3R_bind_caa_f.setKcst(caa_f)
    ip3_IP3R_bind_caa_b.setKcst(caa_b)

    ip3_IP3R_bind_cai_f.setKcst(cai_f)
    ip3_IP3R_bind_cai_b.setKcst(cai_b)
	
    cai_IP3R_bind_ca_f.setKcst(caa_f)
    cai_IP3R_bind_ca_b.setKcst(caa_b)

    open_IP3R_bind_ca_f.setKcst(cai_f)
    open_IP3R_bind_ca_b.setKcst(cai_b)

    unb_IP3R_bind_ip3_f.setKcst(ip3_f)
    unb_IP3R_bind_ip3_b.setKcst(ip3_b)

    caa_IP3R_bind_ip3_f.setKcst(ip3_f)
    caa_IP3R_bind_ip3_b.setKcst(ip3_b)

    cai_IP3R_bind_ip3_f.setKcst(ip3_f)
    cai_IP3R_bind_ip3_b.setKcst(ip3_b)

    cai_ip3_IP3R_bind_ca_f.setKcst(caa_f)
    cai_ip3_IP3R_bind_ca_b.setKcst(caa_b)

    ca2_IP3R_bind_ip3_f.setKcst(ip3_f)
    ca2_IP3R_bind_ip3_b.setKcst(ip3_b)

    # Ca2+ influx through open IP3R channels
    Ca_IP3R_flux.setKcst(6e3)

    return mdl


########################################################################
def gen_geom(mesh):
    # Import the tetrahedral mesh, scale in micrometers (1e-6)
    mesh = meshio.importGmsh(mesh, 1e-6)[0]
    
    # Create a compartment comprising all mesh tetrahedra
    ntets = mesh.countTets()
    tets = range(mesh.ntets)

    # Code to discern triangles belonging to the ER and plasma membranes
    surf_tris = mesh.getSurfTris()
    G = nx.Graph()
    for tri in surf_tris:
        tri_neighbors = mesh.getTriTriNeighbs(tri)
        for neigh_tri in tri_neighbors:
            if neigh_tri in surf_tris:
                G.add_edge(tri, neigh_tri)
    surf_tri_groups = sorted(nx.connected_components(G), key=len, reverse=True)
    cyt_tris = list(surf_tri_groups[0])
    er_tris = diff(surf_tris, cyt_tris)

    # Creation of the cytosolic compartment
    cyto = stetmesh.TmComp('cyto', mesh, range(ntets))
    cyto.addVolsys('vsys')

    # Creation of the ER surface patch 
    er_patch = stetmesh.TmPatch('er_patch', mesh, er_tris, cyto)
    er_patch.addSurfsys('ssys')
    er_area = er_patch.getArea()

    # Creation of the PAP surface patch 
    cyto_patch = stetmesh.TmPatch('cyto_patch', mesh, cyt_tris, icomp=cyto)
    cyto_patch.addSurfsys('mb_surf')
    cyto_area = cyto_patch.getArea()

    # return geometry container object
    return mesh, cyt_tris, er_tris, er_patch, cyto_patch, er_area, cyto_area

