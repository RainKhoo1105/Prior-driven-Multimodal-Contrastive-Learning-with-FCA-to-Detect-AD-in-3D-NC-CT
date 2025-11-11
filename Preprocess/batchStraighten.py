## Usage in 3D Slicer python interactor:
# script_path =r'xx\batchStraighten.py'# path to the script
# exec(open(script_path).read())

import os
import slicer
import vtk
import SimpleITK as sitk
import numpy as np
import CurvedPlanarReformat as CPR

# Define the base directories for input and output
input_base_directory = 'input raw data dir'
output_base_directory = 'output straightened data dir'

# Ensure the output directory exists; if not, create it
if not os.path.exists(output_base_directory):
    os.makedirs(output_base_directory)

# Traverse all subfolders under the input directory
for sub_folder in os.listdir(input_base_directory):
    sub_folder_path = os.path.join(input_base_directory, sub_folder)

    # skip file
    if not os.path.isdir(sub_folder_path):
        continue

    output_sub_folder_path= os.path.join(output_base_directory, sub_folder)
    if os.path.exists(output_sub_folder_path):
        print(f"output subfolder {sub_folder} already exists, skip.")
        continue

    print(f"process subfolder: {sub_folder}")

    # clean scene
    slicer.mrmlScene.Clear(0)


    ct_path = os.path.join(sub_folder_path, 'CT.nii')
    cta_path = os.path.join(sub_folder_path, 'CTA_register.nii')
    tl_path = os.path.join(sub_folder_path, 'TL_register.nii.gz')
    fl_path = os.path.join(sub_folder_path, 'FL_register.nii.gz')
    mask_path=os.path.join(sub_folder_path, 'TotalSeg_TL.nii.gz')


    fl = sitk.ReadImage(fl_path)
    tl = sitk.ReadImage(tl_path)

    fl_array = sitk.GetArrayFromImage(fl)
    tl_array = sitk.GetArrayFromImage(tl)


    # mask = np.logical_or(fl_array > 0, tl_array > 0).astype(np.uint8)


    newmask = np.zeros(fl_array.shape, dtype=np.uint8)
    newmask[tl_array > 0] = 1  # TL
    newmask[fl_array > 0] = 2  # FL

    scene = slicer.mrmlScene


    ct_node = slicer.util.loadVolume(ct_path)
    ijkToRAS = vtk.vtkMatrix4x4()
    ct_node.GetIJKToRASMatrix(ijkToRAS)


    ctm_node = slicer.util.addVolumeFromArray(newmask, ijkToRAS=ijkToRAS, name="ctm", nodeClassName='vtkMRMLLabelMapVolumeNode')
    mask_node = slicer.util.loadSegmentation(mask_path)

    # mask_labelmap = slicer.util.addVolumeFromArray(mask, ijkToRAS=ijkToRAS, name="mask", nodeClassName='vtkMRMLLabelMapVolumeNode')
    # mask_node = scene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    # slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(mask_labelmap, mask_node)
    #
    mask = mask_node.GetAttribute("Segment_1")
    segmentation = mask_node.GetSegmentation()
    segmentID = segmentation.GetSegmentIdBySegmentName('Segment_1')
    mask_node.CreateClosedSurfaceRepresentation()


    extract_centerline_module = slicer.modules.extractcenterline
    extract_centerline_logic = slicer.util.getModuleLogic('ExtractCenterline')


    new_node = scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")


    extractCenterlineWidget = slicer.modules.extractcenterline.widgetRepresentation().self()
    extractCenterlineWidget._parameterNode.SetNodeReferenceID("InputSurface", mask_node.GetID())
    extractCenterlineWidget._parameterNode.SetNodeReferenceID("EndPoints", new_node.GetID())
    extractCenterlineWidget.onAutoDetectEndPoints()

    endpoints = slicer.util.getNode("vtkMRMLMarkupsFiducialNode*")
    print(endpoints.GetNumberOfControlPoints())

    p0 = endpoints.GetNthControlPointPosition(0)
    min_s_value = float('inf')
    min_id = 0
    for i in range(endpoints.GetNumberOfControlPoints()):
        point = endpoints.GetNthControlPointPosition(i)
        s_value = point[2]
        if s_value < min_s_value:
            min_s_value = s_value
            min_id = i

    p1 = endpoints.GetNthControlPointPosition(min_id)

    for i in range(endpoints.GetNumberOfControlPoints() - 1, -1, -1):
        point = endpoints.GetNthControlPointPosition(i)
        if point!= p0 and point!= p1:
            endpoints.RemoveNthControlPoint(i)


    extractCenterlineWidget._parameterNode.SetNodeReferenceID("CenterlineModel", "create new model")
    extractCenterlineWidget._parameterNode.SetNodeReferenceID("CenterlineCurve", "create new curve")
    centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Centerline curve")


    segmentVtkPolyData = vtk.vtkPolyData()  # create empty vtkPolyData object

    mask_node.GetClosedSurfaceRepresentation(segmentID, segmentVtkPolyData)
    centerlinePolyData, voronoiDiagramPolyData = extract_centerline_logic.extractCenterline(segmentVtkPolyData, endpoints)
    centerlinePropertiesTableNode = None
    extract_centerline_logic.createCurveTreeFromCenterline(centerlinePolyData, centerlineCurveNode, centerlinePropertiesTableNode)


    if not os.path.exists(output_sub_folder_path):
        os.makedirs(output_sub_folder_path)


    slicer.util.saveNode(centerlineCurveNode, os.path.join(output_sub_folder_path, f"centerline_{sub_folder}.mrk.json"))


    fieldOfView = [80.0, 80.0]
    outputSpacing = [1.0, 1.0, 5.0]
    logic = CPR.CurvedPlanarReformatLogic()

    straighteningTransformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode', 'Straightening transform')
    logic.computeStraighteningTransform(straighteningTransformNode, centerlineCurveNode, fieldOfView, outputSpacing[2])

    straightenedVolume_ct = slicer.modules.volumes.logic().CloneVolume(ct_node, ct_node.GetName() +'straightened')
    straightenedVolume_ctm = slicer.modules.volumes.logic().CloneVolume(ctm_node, ctm_node.GetName() +'straightened')
    logic.straightenVolume(straightenedVolume_ct, ct_node, outputSpacing, straighteningTransformNode)
    logic.straightenVolume(straightenedVolume_ctm, ctm_node, outputSpacing, straighteningTransformNode)


    slicer.util.saveNode(straighteningTransformNode, os.path.join(output_sub_folder_path, f"straightening_transform_{sub_folder}.tfm"))
    slicer.util.saveNode(straightenedVolume_ct, os.path.join(output_sub_folder_path, f"straightenct_{sub_folder}.nii.gz"))
    slicer.util.saveNode(straightenedVolume_ctm, os.path.join(output_sub_folder_path, f"straightenmask_{sub_folder}.nii.gz"))


    cta_node = slicer.util.loadVolume(cta_path)
    straightenedVolume_cta = slicer.modules.volumes.logic().CloneVolume(cta_node, cta_node.GetName() +'straightened')
    logic.straightenVolume(straightenedVolume_cta, cta_node, outputSpacing, straighteningTransformNode)
    slicer.util.saveNode(straightenedVolume_cta, os.path.join(output_sub_folder_path, f"straightencta_{sub_folder}.nii.gz"))

    print(f"subfolder {sub_folder} processed.")
