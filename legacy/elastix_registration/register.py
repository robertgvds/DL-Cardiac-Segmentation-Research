import itk
import SimpleITK as sitk
import numpy as np
import os

def get_annotated_frame(mask_path):
    """ Encontra o frame 4D com a maior quantidade de pixels anotados. """
    mask_img = sitk.ReadImage(mask_path)
    if mask_img.GetDimension() == 4:
        arr = sitk.GetArrayFromImage(mask_img) 
        # conta pixels > 0 em cada fatia do tempo (t)
        pixels_per_frame = [np.sum(arr[t] > 0) for t in range(arr.shape[0])]
        # retorna o índice do frame mais preenchido
        return int(np.argmax(pixels_per_frame))
    return 0

def extract_3d_volume(image_path, frame_index, temp_suffix):
    """ Fatia o arquivo 4D no frame escolhido e salva como 3D temporário. """
    img = sitk.ReadImage(image_path)
    if img.GetDimension() == 4:
        img_3d = img[:, :, :, frame_index] 
        temp_path = image_path.replace(".nii", f"_{temp_suffix}_temp.nii")
        sitk.WriteImage(img_3d, temp_path)
        return temp_path, True
    return image_path, False

def prepare_full_mask(mask_path, frame_index):
    """ Extrai o frame 3D da máscara mantendo TODOS os rótulos originais (1, 2, 3...). """
    img = sitk.ReadImage(mask_path)
    if img.GetDimension() == 4:
        img_3d = img[:, :, :, frame_index]
    else:
        img_3d = img
        
    temp_path = mask_path.replace(".nii.gz", "_mask_full_temp.nii.gz").replace(".nii", "_mask_full_temp.nii")
    sitk.WriteImage(img_3d, temp_path)
    return temp_path, True

def register_cine_to_lge(cine_path, lge_path, cine_seg_path, output_seg_path):
    print(f"  -> Passo 1: Fatiando imagens 4D para 3D...")
    best_frame = get_annotated_frame(cine_seg_path)
    cine_3d_path, cine_is_temp = extract_3d_volume(cine_path, best_frame, "cine")
    seg_3d_path, seg_is_temp = prepare_full_mask(cine_seg_path, best_frame)

    lge_img = itk.imread(lge_path, itk.F)
    cine_img = itk.imread(cine_3d_path, itk.F)
    cine_mask_float = itk.imread(seg_3d_path, itk.F)

    print(f"  -> Passo 2: Constrói o Elastix...")
    parameter_object = itk.ParameterObject.New()
    
    # ===================================
    # TRANSFORMAÇÃO 1: AFFINE
    # ===================================
    affine_map = parameter_object.GetDefaultParameterMap('affine')
    affine_map['Metric'] = ['NormalizedMutualInformation']
    affine_map['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    
    # Reduz de 6 para 3 resoluções para evitar o achatamento do eixo Z
    affine_map['NumberOfResolutions'] = ['3']
    
    # Protege o eixo Z: (X, Y, Z). 
    # Res 0: Reduz X e Y por 4, Z intacto. Res 1: Reduz X e Y por 2. Res 2: Tamanho original.
    affine_map['ImagePyramidSchedule'] = ['4', '4', '1', '2', '2', '1', '1', '1', '1']
    
    affine_map['MaximumNumberOfIterations'] = ['1000']
    
    # Muda de CenterOfGravity para GeometryCenter (alinha pelo centro das caixas físicas, 
    # ignorando se o sangue preto da LGE puxar o centro de gravidade para o lado errado)
    affine_map['AutomaticTransformInitialization'] = ['true']
    affine_map['AutomaticTransformInitializationMethod'] = ['GeometryCenter']
    
    affine_map['NumberOfSpatialSamples'] = ['10000'] # 10.000 é mais estável para corações
    affine_map['NewSamplesEveryIteration'] = ['true']
    affine_map['ImageSampler'] = ['Random']
    
    parameter_object.AddParameterMap(affine_map)

    # ======================================
    # TRANSFORMAÇAO 2: B-SPLINE
    # ======================================
    # bspline_map = parameter_object.GetDefaultParameterMap('bspline')
    
    # bspline_map['Metric'] = ['NormalizedMutualInformation', 'TransformBendingEnergyPenalty']
    # bspline_map['Metric0Weight'] = ['1.0']
    # bspline_map['Metric1Weight'] = ['5.0']
    
    # bspline_map['NumberOfResolutions'] = ['3']
    # bspline_map['ImagePyramidSchedule'] = ['4', '4', '1', '2', '2', '1', '1', '1', '1']
    
    # # CORREÇÃO: Alinha o espaçamento da malha às 3 resoluções, protegendo a grade no eixo Z
    # bspline_map['GridSpacingSchedule'] = ['4.0', '4.0', '1.0', '2.0', '2.0', '1.0', '1.0', '1.0', '1.0']
    
    # bspline_map['MaximumNumberOfIterations'] = ['1000']
    # bspline_map['FinalGridSpacingInPhysicalUnits'] = ['32.0']
    # bspline_map['FinalBSplineInterpolationOrder'] = ['3']
    
    # parameter_object.AddParameterMap(bspline_map)

    # ===================================
    # EXECUÇÃO DO REGISTRO E TRANSFORMIX
    # ===================================
    print(f"  -> Passo 3: Executa o Registro da Imagem...")
    registered_cine, result_transform_parameters = itk.elastix_registration_method(
        lge_img, cine_img,
        parameter_object=parameter_object,
        log_to_console=True 
    )
    
    debug_path = os.path.join(os.path.dirname(output_seg_path), "cine_registrado_debug.nii")
    itk.imwrite(registered_cine, debug_path)

    print(f"  -> Passo 4: Transfere a máscara completa...")
    # O NearestNeighbor é crucial aqui: impede que o rótulo 1 (Cavidade) e 2 (Miocárdio)
    # se misturem e criem um rótulo "1.5" inexistente na borda.
    result_transform_parameters.SetParameter("ResampleInterpolator", ["FinalNearestNeighborInterpolator"])
    
    lge_full_mask_itk = itk.transformix_filter(
        cine_mask_float,
        transform_parameter_object=result_transform_parameters,
        log_to_console=False
    )
    
    mask_array = itk.array_from_image(lge_full_mask_itk)
    
    # Apenas arredondamos para garantir que são números inteiros, sem binarizar
    mask_array = np.round(mask_array).astype(np.uint8) 
    
    final_mask = itk.image_from_array(mask_array)
    final_mask.CopyInformation(lge_img) 
    
    itk.imwrite(final_mask, output_seg_path)
    print(f"  -> Máscara completa salva em: {output_seg_path}")
    
    if cine_is_temp: os.remove(cine_3d_path)
    if seg_is_temp: os.remove(seg_3d_path)