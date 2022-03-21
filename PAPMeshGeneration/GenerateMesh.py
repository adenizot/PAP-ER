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
SYNAPSE_NAME = 'd1s15a32b1'
WORK_COLLECTION = 'Collection'                            # Collection where PAP and ER object are located
PAP_OBJECT_NAME = 'PM_' + str(SYNAPSE_NAME)               # Name of the PAP object 
COPY_PAP_OBJECT_NAME = 'copy.' + str(PAP_OBJECT_NAME)     # Name of the copy of the PAP object 
ER_OBJECT_NAME = 'ER_' + str(SYNAPSE_NAME)                # Name of the ER object 
ERPM_dist = 0.02                                          # Threshold distance defining of ER-plasma membrane contact site
ERPM_TEXT_FILENAME = 'ERPMd_' + str(SYNAPSE_NAME) + '_fr' # Name of the file in which the distance between each PM vertex and the closest ER vertex is stored
STL_FILENAME = str(SYNAPSE_NAME) + '_fr'                   # Name of the stl file exported

# ER SPLITTING PARAMETERS
COLL_MARGIN_PM = 0.005          # Affects how close ER objects can get to the PAP plasma membrane. Larger values may prevent intersections 
                 		 # However, if the value is too large, it may send the collided ER objects far out of the cell 
COLL_MARGIN_ER = 0.001          # Affects how close ER objects can get. Larger values may prevent intersections
SPLIT_CUBE_SIZE = 0.1           # Size of the cubes used to split the original ER object into smaller ER objects 
VOL_TOL = 0.00003               # Volume tolerance/threshold below which the object is deleted because it is too small
DEBUG_SPLIT = False             #If true, we don't run physics and stop only after ER splitted. 

# ER SCATTERING PARAMETERS
N_FRAMES = 100                  # How many frames physics simulation will be running. Each frame can then be exported as an .stl file (see line )
PHYS_COLLECTION = 'Physics'     # This collection will be created for physics simulation
PHYS_CONSTRAINTS = 'constraints'# Collection that will be created and used by physics and to store a 'Force' object

# Note: remember to hide from the viewport other objects than the new ER and PAP objects in order to visualize the movement of the ER objects between each frame of the physics simulation

#######################
###### FUNCTIONS ######
#######################
# Calculate the surface area of a mesh
def bmesh_calc_area(bm):
    return sum(f.calc_area() for f in bm.faces)


# Returns a transformed, triangulated copy of the mesh as BMesh object
def bmesh_copy_from_object(obj, apply_modifiers=False):
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
    bm.transform(obj.matrix_world)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    return bm 
    
      
# Creates a copy of the object (and mesh data)    
def CopyObjectAndData(coll, obj):    
    newObj = obj.copy();                        # copy base object
    newObj.data = obj.data.copy();              # copy mesh data too 
    newObj.location = obj.location              # set the position
    newObj.name = 'copy.' + obj.name 
    coll.objects.link(newObj)
    return newObj 


# Writes data (list of strings) in a text file
def WriteTXTOutput(listdata,filename):
    with open(filename, 'w') as f:
        sdata = '\n'.join(listdata)
        f.writelines(sdata)
                     

# For visualization purposes (used to visualize ER-PM contact sites) - creates a mesh made of vertices
def CreatePointCloudObject(points):
    # make mesh
    vertices = points
    edges = []
    faces = []
    new_mesh = bpy.data.meshes.new('new_mesh')
    new_mesh.from_pydata(vertices, edges, faces)
    new_mesh.update()
    # make object from mesh
    new_object = bpy.data.objects.new('new_object', new_mesh)
    # make collection
    new_collection = bpy.data.collections.new('new_collection')
    bpy.context.scene.collection.children.link(new_collection)
    # add object to scene collection
    new_collection.objects.link(new_object)


# Creates a KDtree for vertices in bmesh
def CreateKDTree(bm):
    oKDTree = mathutils.kdtree.KDTree(len(bm.verts))
    for i, v in enumerate(bm.verts):
        oKDTree.insert(v.co, i)
    oKDTree.balance()
    return oKDTree


# Returns a list of vertices from 2 meshes within a given distance
def GetVertsInDistance(bm1,bm2,val):
    # Use blender's KDTrees for faster search
    kd1 = CreateKDTree(bm1)
    kd2 = CreateKDTree(bm2)
    # Store vertex indices within the distance 
    # May have duplicated vertices
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
    

# Returns true if the face is selected (all the vertices belonging to the face should be selected)    
def isSelected(face):
    selected = [v for v in face.verts if v.select == True]
    return len(selected) == len(face.verts)


# Get faces to which the vertices belong
def ComputeAreaForVerts(obj, bm, inds):
    print(obj,obj.name)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(state=True)    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')  # deselect all data in edit mode
    bpy.ops.object.mode_set(mode='OBJECT')
    bm.verts.ensure_lookup_table()
    for i in inds:
        obj.data.vertices[i].select = True
        bm.verts[i].select = True 
    return sum(f.calc_area() for f in bm.faces if isSelected(f))


# Get the number of PM and ER vertices within a given distance (here ER-PM contact sites: 20 nm
def GetContactVerticesWithin(oPM, oER, ERPM_dist):
    bpy.ops.object.select_all(action='DESELECT')
    # select the objects
    oPM.select_set(state=True)
    oER.select_set(state=True)
    # get the bmeshes
    bmPM = bmesh_copy_from_object(oPM, apply_modifiers=True) 
    bmER = bmesh_copy_from_object(oER, apply_modifiers=True)
    # Get indices of the vertices that are within the threshold distance
    vPM, vER, indsPM, indsER = GetVertsInDistance(bmPM, bmER, ERPM_dist)
    nbERvertERPMcont = len(indsER)
    nbPMvertERPMcont = len(indsPM)
    print('PM vertices at ER-PM contact sites:',nbPMvertERPMcont)
    print('ER vertices at ER-PM contact sites:',nbERvertERPMcont)
    # Create an object that contains the vertices - Uncomment to visualize contact sites
    #points = []
    #points.extend(vPM)
    #points.extend(vER)
    #CreatePointCloudObject(points)  
    return [nbPMvertERPMcont, nbERvertERPMcont]        

# Returns a list of distances between each PM vertex & the closest ER vertex      
def GetMinDistances(oPM, oER):
    d = set() #set of min distances for each vertex in PM to ER
    bmPM = bmesh_copy_from_object(oPM, apply_modifiers=True) 
    bmER = bmesh_copy_from_object(oER, apply_modifiers=True)
    kdER = CreateKDTree(bmER)
    for ibm, vbm in enumerate(bmPM.verts):
        # Find the nearest vertex in ER to vbm
        ver, ier, distPM_to_ER = kdER.find(vbm.co)    
        d.add(distPM_to_ER)
    return list(d)

# Rescales an object to a given scale
def ScaleObjectInPlace(obj, scale):
    bbox = [Vector(b) for b in obj.bound_box]
    o = sum(bbox, Vector()) / 8
    T = Matrix.Translation(o)
    S = Matrix.Diagonal(scale).to_4x4()
    T2 = Matrix.Translation(-o)
    M = T @ S @ T2
    obj.data.transform(M)
    obj.data.update()


# Rescales the size of split ER objects so that the sum of their surface area equals the surface area of the original ER object
def MatchNewERPartsToOriginalArea(originER, partsColl, newPM):
    print('---------MatchNewERPartsToOriginalArea----------')
    epsilon = 0.0001
    bm = bmesh_copy_from_object(originER, apply_modifiers=True)
    # Get area of the original ER object
    originArea = bmesh_calc_area(bm)
    originVolume = bm.calc_volume() 
    bm.free()
    # Get a list of all ER objects created after ER splitting
    ERparts = [o for o in partsColl.objects if o.name != newPM.name]    
    # Iteratively rescale ER objects until the sum of their area reaches the area of the original ER object
    for n in range(0,10):
        areaParts = 0
        volParts = 0
        for oER in ERparts: 
            bm = bmesh_copy_from_object(oER, apply_modifiers=True)
            area = bmesh_calc_area(bm)      
            areaParts += area
            volParts += bm.calc_volume() 
            bm.free()  
        diff = originArea - areaParts    
        scale = (originArea/areaParts) ** (2/3)
        print(originArea, areaParts, diff, scale)
        if abs(diff) < epsilon:
            break
        for oER in ERparts: 
            ScaleObjectInPlace(oER, (scale,scale,scale))   


# Prepare ER and PAP objects to be exported in .stl format
def GetProcessAtFrame(collection_physics, oPM, nFrame):
    bpy.context.scene.frame_set(nFrame)                         # set the frame
    STL_Collection = bpy.data.collections.new('STL.Frame' + str(bpy.context.scene.frame_current).zfill(3))            # make collection
    bpy.context.scene.collection.children.link(STL_Collection)  # link collection to the scene 
    bmERs = []
    for er in collection_physics.objects:
        if er == oPM:
            continue
        erBM = bmesh_copy_from_object(er, apply_modifiers=True)
        for f in erBM.faces:
            f.normal_flip()
        bmERs.append(erBM)
        ermesh = bpy.data.meshes.new('ermesh')
        erBM.to_mesh(ermesh)
        erBM.free()
        erOb = bpy.data.objects.new('stlER.000', ermesh)        # make object from mesh
        STL_Collection.objects.link(erOb)                       # add object to scene collection                     
    new_mesh = bpy.data.meshes.new('new_mesh')
    bmPM = bmesh_copy_from_object(oPM, apply_modifiers=False)  # Create a PAP bmesh without modifiers so that physics only applies to ER objects
    bmPM.to_mesh(new_mesh)
    bmPM.free()
    stlPM = bpy.data.objects.new('stlPM', new_mesh)             # make object from mesh
    STL_Collection.objects.link(stlPM)                          # add object to scene collection    
    bpy.ops.object.select_all(action='DESELECT')                # deselect all objects
    # join all objects in the collection
    for o in STL_Collection.objects:
        o.select_set(state=True)
    stlER = STL_Collection.objects[0]
    bpy.context.view_layer.objects.active = stlER
    stlER.name = 'stl_ER.Fr.' + str(bpy.context.scene.frame_current).zfill(3)
    stlPM.select_set(state=False)
    stlPM.name = 'stl_PM.Fr.'+ str(bpy.context.scene.frame_current).zfill(3)
    bpy.ops.object.join()
    return STL_Collection, stlPM, stlER


# Get the frame from the physics collection and export to STL format
# 3D meshing of the mesh can be performed using TetWild:
# https://github.com/Yixin-Hu/TetWild/blob/master/README.md
def ExportSTL(Coll, oPM, oER):
    copyPM = CopyObjectAndData(Coll,oPM)
    copyER = CopyObjectAndData(Coll, oER)
    bpy.ops.object.select_all(action='DESELECT')
    copyER.select_set(state=True)
    copyPM.select_set(state=True)
    bpy.context.view_layer.objects.active = copyPM
    bpy.ops.object.join()
    bpy.ops.export_mesh.stl(filepath='stl.frame.'+str(bpy.context.scene.frame_current).zfill(3)+'.stl', use_selection=True) #export mesh
    bpy.ops.object.delete()
    

# Run Physics simulation
def RunPhysics(collection, oPM, frames):
    # Add rigid body world (if not already in the scene)
    if bpy.context.scene.rigidbody_world == None:
        bpy.ops.rigidbody.world_add()
    bpy.context.scene.rigidbody_world.enabled = True
    bpy.context.scene.use_gravity = False
    bpy.context.scene.rigidbody_world.time_scale = 0.1                      # Parameter that alters the speed of ER movement between each frame
    bpy.context.scene.rigidbody_world.substeps_per_frame = 50               # Number of steps to compute motions - larger values reduce object intersections
    bpy.context.scene.rigidbody_world.collection = collection
    constraintsColl = bpy.data.collections.new(PHYS_CONSTRAINTS)            # Make collection
    bpy.context.scene.collection.children.link(constraintsColl)             # Link collection to the scene
    bpy.context.scene.rigidbody_world.constraints = constraintsColl
    bpy.ops.object.select_all(action='DESELECT')
    for o in collection.objects:
        o.select_set(state = True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.rigidbody.object_add()
        o.rigid_body.collision_shape = 'MESH'
        o.rigid_body.mesh_source = 'FINAL'
        o.rigid_body.restitution = 0.1                    
        o.rigid_body.mass = 0.1
        o.rigid_body.collision_margin = COLL_MARGIN_ER       # Can be altered to decrease collision between ER objects
    oPM.rigid_body.collision_margin = COLL_MARGIN_PM         # Can be altered to decrease collision between ER and PAP objects
    bpy.context.view_layer.objects.active = oPM  
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.ops.object.effector_add(type='FORCE', radius=0.5, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    force = bpy.context.object
    bpy.ops.collection.objects_remove_all()
    constraintsColl.objects.link(force)
    force.field.strength = 0.1
    bpy.context.scene.frame_end = frames
    bpy.ops.ptcache.free_bake_all()
    bpy.context.scene.rigidbody_world.point_cache.frame_end=bpy.context.scene.frame_end
    bpy.ops.ptcache.bake_all(bake=True)


def ClearParent(oObj):    
    bpy.context.view_layer.objects.active = oObj
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')


# Move objects to the origin
def ProcessToOrigin(sCollection, pm, er):
    # create a working collection
    workCollection = bpy.data.collections.new(sCollection)                  # make collection
    bpy.context.scene.collection.children.link(workCollection)              # link newly created collection to the scene
    # get PAP and ER objects
    oPM = bpy.data.objects[pm]
    ClearParent(oPM)
    # unparent any children
    for child in bpy.data.objects[pm].children:
        child.hide_viewport = False
        child.select_set(state=True)
        ClearParent(child) 
    for child in bpy.data.objects[er].children:
        child.hide_viewport = False
        child.select_set(state=True)
        ClearParent(child) 
    oER = bpy.data.objects[er]
    # add PAP and ER objects to the working collection
    workCollection.objects.link(oPM) 
    workCollection.objects.link(oER) 
    for o in bpy.data.collections[sCollection].objects:
        o.select_set(state=True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    # parent ER to PM
    oPM.select_set(state=False)
    bpy.context.view_layer.objects.active = oPM
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    # move PM to origin (er moves, too because it's a child of PM transform-wise)
    oPM.location = [0,0,0]   
    # unparent ER from PM and keep transform    
    ClearParent(oER)
    return oPM, oER


# Splits the ER into smaller ER objects
def SplitER(oColl, oER, oPM, cubeSize, voltolerance):
    x,y,z = oER.location.x - oER.dimensions.x/2 +cubeSize/2, oER.location.y - oER.dimensions.y/2 +cubeSize/2 , oER.location.z - oER.dimensions.z/2 +cubeSize/2
    bpy.ops.mesh.primitive_cube_add(size=cubeSize, align='WORLD', location=(x, y, z))
    cube = bpy.context.object
    bpy.ops.collection.objects_remove_all()
    oColl.objects.link(cube) 
    # Create Cube Copies
    countX = math.ceil(oER.dimensions.x / cubeSize) + 1
    countY = math.ceil(oER.dimensions.y / cubeSize) + 1
    countZ = math.ceil(oER.dimensions.z / cubeSize) + 1
    print(countX,countY,countZ, 'Total:', countX * countY * countZ )
    newERsCubes = []
    for x in range(0, countX):
        for y in range(0, countY):
            for z in range(0, countZ):
                o = CopyObjectAndData(oColl,cube)
                o.location = cube.location + Vector((x*cubeSize,y*cubeSize,z*cubeSize))
                newERsCubes.append(o)
    bpy.ops.object.select_all(action='DESELECT')
    cube.select_set(state=True)  
    bpy.ops.object.delete()                 # delete the first original cube
    # For each cube, apply intersect boolean between ER and cube
    for newER in newERsCubes:
        newER.modifiers.new('boolean1','BOOLEAN')
        newER.modifiers['boolean1'].object = oER               # set the base volume
        newER.modifiers['boolean1'].operation = 'INTERSECT'     
        newER.select_set(state=True)
        bpy.context.view_layer.objects.active = newER
        bpy.ops.object.modifier_apply(modifier='boolean1')
        newER.select_set(state=False)         
        print('processed.',newER.name)       
        newER.name = 'ER3.000'
    bpy.ops.object.select_all(action='DESELECT') 
    # Remove empty objects (number of vertices <= 3) 
    for o in oColl.objects:
        if len(o.data.vertices) < 4:
            o.select_set(state = True)
        obm = bmesh_copy_from_object(o, apply_modifiers=False)  
        ovol = obm.calc_volume() 
        print('vol:', ovol, '\n')
        obm.free()
        if ovol < voltolerance:
             o.select_set(state = True)
    bpy.ops.object.delete()   # delete all empty meshes       
    # now test if there are any open loops or loose vertices (merge by distance and look for empty meshes again)
    # join the rest of the ER (except for oER), forming a new ER object
    bpy.ops.object.select_all(action='DESELECT') 
    oER.select_set(state = True)
    bpy.ops.object.delete()                     # delete old oER object
    newPM = CopyObjectAndData(oColl, oPM)
    newPM.modifiers.new('solidify','SOLIDIFY')
    newPM.modifiers["solidify"].thickness = -0.02
    return newPM


'''
------------------------------------------------------------------------------------------
                        Main Pipeline functions: 
------------------------------------------------------------------------------------------
'''
# Setup the scene and objects. Splits ER into smaller parts 
def RunSplitter():
    # (1) place ER and PAP objects at the origin
    originalPM, originalER = ProcessToOrigin(sCollection = WORK_COLLECTION, pm = PAP_OBJECT_NAME, er = ER_OBJECT_NAME)
    
    # (2) Clone objects to the new collection for ER splitting
    phyColl = bpy.data.collections.new(PHYS_COLLECTION)         # make collection
    bpy.context.scene.collection.children.link(phyColl)         # link collection to the scene 
    splitER = CopyObjectAndData(phyColl,originalER)             # create a new ER object for splitting

    # (3) Split the ER 
    oPM = SplitER(oColl = phyColl, oER = splitER, oPM = originalPM, cubeSize = SPLIT_CUBE_SIZE,voltolerance = VOL_TOL)

    # (4) Rescale ER objects to get the same surface area as the original ER
    oPM = bpy.data.objects['copy.' + PAP_OBJECT_NAME]
    MatchNewERPartsToOriginalArea(originalER, phyColl, oPM)


# Scatter the split ER objects 
def RunScatter():    
    if DEBUG_SPLIT == True:
        print('You are running in debug splitter mode, \n To Run scatter, set DEBUG_SPLIT to False')
        return
    oPM = bpy.data.objects['copy.' + PAP_OBJECT_NAME]
    phyColl = bpy.data.collections[PHYS_COLLECTION]
    #(5)update physical RigidBody properties
    RunPhysics(phyColl, oPM, N_FRAMES)


# Analyses ER-PM contact sites of the selected frame and exports the PAP mesh in stl format. Can be run on all frames of the physics simulation if desired. 
def SaveSTLandMeasureERPMForFrame(frame = 1):
    oPM = bpy.data.objects['copy.' + PAP_OBJECT_NAME]
    phyColl = bpy.data.collections[PHYS_COLLECTION]
    # Prepare ER and PAP objects for mesh export
    stlColl, stlPM, stlER = GetProcessAtFrame(phyColl,oPM,frame)
    # Export the PAP mesh from the current frame to stl format
    ExportSTL(stlColl, stlPM, stlER)

    # Get number of PM and ER vertices at ER-PM contact sites
    PM_contvert, ER_contvert = GetContactVerticesWithin(stlPM, stlER, ERPM_dist)
    
    # Measures the distance between each PM vertex and the closest ER vertex
    d = GetMinDistances(stlPM, stlER)
    WriteTXTOutput([str(x) for x in d],TEXT_FILENAME+str(bpy.context.scene.frame_current).zfill(3)+'.txt')


'''
------------------------------------------------------------------------------------------
                                Script Entry point.
                        Computes ER-PM distance and exports STL for the nFrame
------------------------------------------------------------------------------------------
'''
# Split the ER. Results can be altered by scatter.
# ! Careful: run only once and then comment out. If unsatisfied with the result, delete old ER objects and repeat with different parameter values l. 24-28
RunSplitter()
    
# Run physics simulation.
# ! Careful: run only once and then comment out. If unsatisfied with the result, try with different parameter values (l. 32-34) on the original ER objects
RunScatter()

if DEBUG_SPLIT == False:
    # Measure ER-PM contact sites and save frames as STL meshes for simulations
    #for n in range(0, N_FRAMES):
    #    SaveSTLandMeasureERPMForFrame(frame = n)

    #OR, Compute for specific frame:
    SaveSTLandMeasureERPMForFrame(frame = 10)

