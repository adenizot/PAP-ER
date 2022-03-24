import sys
import os
import traceback
import bpy
import numpy as np
import mathutils
import math
import time
from mathutils import Matrix, Vector
import bmesh


########################
### PARAMETER VALUES ###
########################
WORK_COLLECTION = 'Collection 1' # Name of the collection that contains the objects
PAP_OBJECT_NAME = 'PAP'          # Name of the PAP object 
ER_OBJECT_NAME = 'ER'            # Name of the ER object 
PSD_OBJECT_NAME = 'd1s15a32b1_E' # Name of the PSD object 
ERPM_dist = 0.02                 # Threshold distance defining ER-PM contact site (in micrometers)


#######################
###### FUNCTIONS ######
#######################

# Calculates the surface area of an object
def bmesh_calc_area(bm):
    return sum(f.calc_area() for f in bm.faces)


# Writes data (list of strings) in a text file
def WriteTXTOutput(listdata,filename):
    with open(filename, 'w') as f:
        sdata = '\n'.join(listdata)
        f.writelines(sdata)
                     
                     
# Returns a transformed, triangulated copy of a mesh
def bmesh_copy_from_object(obj, transform=True, triangulate=True, apply_modifiers=False):
    assert(obj.type == 'MESH')
    if apply_modifiers and obj.modifiers:
        import bpy
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        me = obj_eval.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(me)
        obj_eval.to_mesh_clear()
        del bpy
    else:
        me = obj.data
        if obj.mode == 'EDIT':
            bm_orig = bmesh.from_edit_mesh(me)
            bm = bm_orig.copy()
        else:
            bm = bmesh.new()
            bm.from_mesh(me)
    if transform:
        bm.transform(obj.matrix_world)
    if triangulate:
        bmesh.ops.triangulate(bm, faces=bm.faces)
    return bm   

# Creates a point cloud to visualize vertices stored in 'points'
def CreatePointCloudObject(points):
    # Create a new mesh
    vertices = points
    edges = []
    faces = []
    new_mesh = bpy.data.meshes.new('new_mesh')
    new_mesh.from_pydata(vertices, edges, faces)
    new_mesh.update()
    # Create an object from the mesh
    new_object = bpy.data.objects.new('new_object', new_mesh)
    # Create a new collection
    new_collection = bpy.data.collections.new('new_collection')
    bpy.context.scene.collection.children.link(new_collection)
    # Add object to the new collection
    new_collection.objects.link(new_object)


# Creates a KDtree for vertices in the mesh
def CreateKDTree(bm):
    oKDTree = mathutils.kdtree.KDTree(len(bm.verts))
    for i, v in enumerate(bm.verts):
        oKDTree.insert(v.co, i)
    oKDTree.balance()
    return oKDTree

# get and visualize ER and PM vertices at ER-PM contact sites
def GetContactVert(oPM, oER, val=ERPM_dist):
    bpy.ops.object.select_all(action='DESELECT')
    #select the objects
    oPM.select_set(state=True)
    oER.select_set(state=True)
    #get the bmeshes
    bmPM = bmesh_copy_from_object(oPM, apply_modifiers=True) 
    bmER = bmesh_copy_from_object(oER, apply_modifiers=True)
    #just test the area:
    #print(bmesh_calc_area(bmPM), bmesh_calc_area(bmER))
    #get indicis of the vertices which are within the threshold value
    vPM, vER, indsPM, indsER = GetVertsInDistance(bmPM, bmER, val)
    print('Number PM vertices at ER-PM contact sites',len(indsPM))
    print('Number ER vertices at ER-PM contact sites',len(indsER))
    #create object out of the vertices
    points = []
    points.extend(vPM)
    points.extend(vER)
    CreatePointCloudObject(points)  
    return [indsPM, indsER]   

# Returns a list of vertices from 2 meshes within a given distance
def GetVertsInDistance(bm1,bm2,val):
    # Use Blender's KDTrees for a faster search
    kd1 = CreateKDTree(bm1)
    kd2 = CreateKDTree(bm2)
    # Store vertex indices within the distance 
    verts1 = []
    verts2 = [] 
    inds1 = set()
    inds2 = set()
    for i1, v1 in enumerate(bm1.verts):
        # Find the nearest point in kd2 to v1.co
        v2co, i2, dist1_to_2 = kd2.find(v1.co)
        if dist1_to_2 < val:
            verts2.append(v2co)
            inds1.add(i1)
            inds2.add(i2)
    for i2, v2 in enumerate(bm2.verts):
        # Find the nearest point in kd1 to v2.co
        v1co, i1, dist2_to_1 = kd1.find(v2.co)
        if dist2_to_1 < val:
            verts1.append(v1co)
            inds1.add(i1)
            inds2.add(i2)            
    return verts1, verts2, inds1, inds2

    
def isSelected(face):
    selected = [v for v in face.verts if v.select == True]
    return len(selected) == len(face.verts)

# Get faces to which vertices belong
def GetObjectForVerts(obj, bm, inds):
    print(obj,obj.name)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(state=True)    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT') #deselect all data in edit mode
    bpy.ops.object.mode_set(mode='OBJECT')
    bm.verts.ensure_lookup_table()
    for i in inds:
        obj.data.vertices[i].select = True
        bm.verts[i].select = True 
    return sum(f.calc_area() for f in bm.faces if isSelected(f))


# Creates a copy of an object    
def CopyObjectAndData(coll, obj):    
    newObj = obj.copy();                        # copy base object
    newObj.data = obj.data.copy();              # copy mesh data too 
    newObj.location = obj.location              # set the position, parameter pos   
    newObj.name = 'copy.' + obj.name 
    coll.objects.link(newObj)
    return newObj           
  
       
# Returns the list of the distance between each PM vertex & the closest ER vertex
def GetMinDistances(oPM, oER):
    d = set() 
    bmPM = bmesh_copy_from_object(oPM, apply_modifiers=True) 
    bmER = bmesh_copy_from_object(oER, apply_modifiers=True)
    kdER = CreateKDTree(bmER)
    for ibm, vbm in enumerate(bmPM.verts):
    # Find the closest ER vertex to vbm
        ver, ier, distPM_to_ER = kdER.find(vbm.co)    
        d.add(distPM_to_ER)
    return list(d)
            


# Center objects to the origin
def ProcessToOrigin(collection, pm, er, psd):
    for o in bpy.data.collections[collection].objects:
        o.select_set(state=True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    oPM = bpy.data.objects[pm]
    oER = bpy.data.objects[er]
    oPSD = bpy.data.objects[psd]
    # parent ER to PM
    oPM.select_set(state=False)
    bpy.context.view_layer.objects.active = oPM
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    oPM.location = [0,0,0]   # move PM to origin
    # unparent ER from PM and keep transform
    bpy.context.view_layer.objects.active = oER
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    return oPM, oER, oPSD



'''
-------------------------------------------------------------------
     To measure and export PM-PSD, ER-PSD and ER-PM distances
-------------------------------------------------------------------
'''
# Move objects to origin
originalPM, originalER, originalPSD = ProcessToOrigin(collection = WORK_COLLECTION, pm =  PAP_OBJECT_NAME, er = ER_OBJECT_NAME, psd=PSD_OBJECT_NAME)

# Measure and record the distance between each vertex on the plasma membrane and the ER
d = GetMinDistances(originalPM, originalER)
WriteTXTOutput([str(x) for x in d],'dist_ERPM_'+PSD_OBJECT_NAME+'.txt')


# Measure and record the distance between each vertex on the plasma membrane and the PSD
d = GetMinDistances(originalPSD, originalPM)
WriteTXTOutput([str(x) for x in d],'dist_PMPSD_'+PSD_OBJECT_NAME+'.txt')

# Measure and record the distance between each vertex on the ER and the PSD
d = GetMinDistances(originalPSD, originalER)
WriteTXTOutput([str(x) for x in d],'dist_ERPSD_'+PSD_OBJECT_NAME+'.txt')

PMcvert, ERcvert = GetContactVert(originalPM, originalER, ERPM_dist)
