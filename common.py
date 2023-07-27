import os 
import numpy as np
import numpy
import cv2 
from deepface import DeepFace
from retinaface import RetinaFace 
from scipy import spatial
import shutil
from deepface.commons import functions, realtime, distance as dst
import ast
import torch
from omegaconf import OmegaConf
from facetorch import FaceAnalyzer
from torch.nn.functional import cosine_similarity
import time
import json
import shutil
from tqdm import tqdm



def plot_one_box(x:list, img:numpy.array, color:list=None, label:str=None, line_thickness:int=None) -> None:
    """
    Function that draw bbox on the image.

    Arguments:
    ------------------------
    x : list
        List with coordinates of a bounding box in format : [x_left_upper, y_left_upper, x_right_bottom, y_right_bottom].
    img : numpy.array
        Image where we need to draw a bounding box.
    color : list : optional
        List that represents color in RGB format. If it is None color will be generated randomly.
    label : str
        Name that is needed to be assigned to a bounding box.
    line_thickness : int
        Integer number that represent the thickness of a bounding box.
    
    Returns: 
    --------------------
    None
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_one_keypoint(x:list, img:numpy.array, color:list, thickness:int, radius:int) -> None:
    """
    Function that draws a keypoint on the image.

    Arguments:
    --------------
    x : list
        Coordinate of a keypoint - [x,y]
    img : numpy.array
        Image where we need to draw a keypoint.
    color : list : optional
        List that represents color in RGB format. If it is None color will be generated randomly.
    thickness : int
        Integer number that represents the thickness of a circle. If thickness is equal to -1 circle will be drawn as point.
    radius : int
        Integer number that represents the radius of a circle. 

    Returns:
    --------------
    None
    """
    tl = thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    cv2.circle(img, x, radius=radius, color=color, thickness=tl)

  

def retrieve_data_from_dict_retinaface(output_dict:dict) -> list:
    """
    Function that converts output from retinaface model into format = [x_top_left, y_top_left, x_bottom_right, y_bottom_left, confidence].

    Arguments:
    --------------
    output_dict : dict
        Dictionary with output of the model.
    Returns:
    --------------
    list
        List of lists with converted detections.
    """
    detections = []
    if str(type(output_dict)) == "<class 'dict'>":
        for key in output_dict.keys():
            identity  = output_dict[key]
            raw_bbox =  identity['facial_area']
            conf = identity['score']
            converted_bbox = [raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3], conf]
            detections.append(converted_bbox)
    return detections

def retrieve_data_from_dict_retinaface_tracker(output_dict:dict):
    """
    Function that converts output from retinaface model into format that can be used in different trackers = [x_top_left, y_top_left, width, height].

    Arguments:
    --------------
    output_dict : dict
        Dictionary with output of the model.
    Returns:
    --------------
    list
        List of lists with converted detections.
    list
        List with confidences for each detections.
    list
        List with id of a class for each object (in our case we have only one class).
    """
    detections = []
    confidences = []
    class_ids = []
    if str(type(output_dict)) == "<class 'dict'>":
        for key in output_dict.keys():
            identity  = output_dict[key]
            raw_bbox =  identity['facial_area']
            conf = identity['score']
            if raw_bbox[2] - raw_bbox[0] < 25 or raw_bbox[3] - raw_bbox[1]<25:
                        continue
            confidences.append(conf)
            class_ids.append(0)
            converted_bbox = np.array([raw_bbox[0], raw_bbox[1], raw_bbox[2] - raw_bbox[0], raw_bbox[3] - raw_bbox[1]])
            detections.append(converted_bbox)
    detections = np.array(detections)
    confidences = np.array(confidences)
    class_ids = np.array(class_ids)
    
    return detections, confidences, class_ids






def convert_frame_to_time(frame_id:int, fps:int, format_of_out:str = 'dash') -> str:
    """
    Function that convert frame number into time in format: mm:ss:ms or mm-ss-ms.

    Arguments:
    ---------------
    frame_id : int
        Frame number which is needed to be converted into time format.
    fps : int
        Amount of frames per second that video has.
    format_of_out : str
        In which format we need to get time. Possible options: ['dash', 'colon']
        Unfortunately os library does not support colon format, so to avoid errors it is better to use dash format.
    
    Returns:
    --------------
    str 
        A string that represent frame number as time.
    """
    amount_of_seconds = frame_id/fps
    # print(amount_of_seconds)
    amount_of_mins = amount_of_seconds/60

    mins_to_print = str(int(amount_of_mins))
    seconds_to_print = str(int(amount_of_seconds%60))
    # print(amount_of_seconds%60)
    miliseconds_to_print = str(amount_of_seconds-int(amount_of_seconds))[2:5]
    # print(miliseconds_to_print)
    if len(seconds_to_print)==1:
        seconds_to_print = '0'+seconds_to_print
    if len(mins_to_print) == 1:
        mins_to_print = '0'+mins_to_print
    if format_of_out == 'colon':
        final_print = mins_to_print+':'+seconds_to_print+':'+miliseconds_to_print
    elif format_of_out == 'dash':
        final_print = mins_to_print+'-'+seconds_to_print+'-'+miliseconds_to_print
    return final_print


# model_name = [VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Ensemble]
def reindetificate(img_1:numpy.array, img_2:numpy.array, threshold:float = 0.5, model_name:str = 'DeepFace',distance_metric:str = 'cosine',normalization:str = 'base', use_default_threshold:bool = True ) -> bool:
    """
    Function that process two images to check their similiarity.

    Arguments:
    --------------------
    img_1 : numpy.array
        First image for check.
    img_2 : numpy.array
        Second image for check.
    threshold : float
        Threshold that defines if images are similiar or not.
    model_name : str
        Name of the model to use. Possible options: [VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Ensemble]
    distance_metric : str
        Which distance metric to use with model. Possible options: ['cosine','euclidean','euclidean_l2']
    normalization : str
        Type of normalization to apply on input image. Possible options: ['base', 'raw', 'Facenet', 'Facenet2018','VGGFace','VGGFace2','ArcFace'].
        For more details please visit page of repository: https://github.com/serengil/deepface/blob/6c5c3511f4f569b74bfe4561b458149bbc6aa5ee/deepface/commons/functions.py#L186
    use_default_threshold : bool
        To use default threshold which was defined by DeepFace or not. 

    Returns:
    ---------------------
    bool
        a boolean variable that shows if images are similiar ot not.
    """
    result = DeepFace.verify(img_1, img_2, model_name = model_name, enforce_detection=False, distance_metric=distance_metric, normalization=normalization)
    distance = result['distance']
    if use_default_threshold == True:
        threshold = result['threshold']
    if distance <= threshold:
        return True
    elif distance > threshold:
        return False


def get_centroid(bbox:list) -> list:
    """
    Function that converts bounding box into list with coordinates of center of this bounding box.

    Arguments:
    -------------
    bbox:list
        List with coordinates of a bounding box - [x_top_left, y_top_left, x_bottom_right, y_bottom_right].
    Returns:
    -------------
    list
        List with coordinates of center of bounding box - [x_cent, y_cent]
    """
    x_cent = int((bbox[2] - bbox[0])/2)
    y_cent = int((bbox[3] - bbox[1])/2)
    return [x_cent, y_cent]

def is_blur(detect:list = None, frame:numpy.array = None, img:numpy.array = None, threshold:int = 100) -> bool:
    """
    Function that check if image is blurry or not.
    
    Arguments:
    --------------
    detect : list : optional
        List with cordinates of a bounding box.
    frame : numpy.array : optional 
        Numpy array that represents whole frame.
    img : numpy.array : optional
        Numpy  array that represents image of a face.
    threshold : int : default=100
        Integer number that defines threshold which separate blurry and not blurry images.
    Important note: to call this function you have to pass detect and frame or img.
    
    Returns: 
    ---------------
    bool
        Boolean variable that shows if image is blurry or not. True -> blurry 
    """
    if img is None:
        image_to_check = frame[detect[1]:detect[3], detect[0]:detect[2]]
        image_to_check_gray = cv2.cvtColor(image_to_check, cv2.COLOR_RGB2GRAY)
    if img is not None:
        image_to_check_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    value = cv2.Laplacian(image_to_check_gray, cv2.CV_64F).var()
    print(value)
    if value < threshold:
        return True
    if value > threshold:
        return False



def send_to_database(path_to_database:str, video_name:str, detect:list, frame:numpy.array, frame_id:int, fps:int, tr_id:int,\
     model_name:str, normalization:str,threshold:float = 1.13, threshold_for_blur:int = 100, use_resize:bool = False, dict_with_embeddings = {}, distance_metric = 'cosine'):
    """
    Function that manage the database. 
    Here is basic pipeline:
    1. Function get the first detect, if base folder of the video is exist move to the next step, if not creates it.
    2. If there is no saved images. Create new folder with id of a detect and saves image of a face into it.
    3. If there are saved images iterate over each of them and count the similiarity with input image (if it isn't blurry).
    4. Finds folder with the biggest amount of similiar images and save input image into this folder.

    Arguments:
    ---------------
    path_to_database : str
        Name of the root dir, the database itself.
    video_name : str
        Name of the video to process.
    detect : list
        List with coordinates of bounding box.
    frame : numpy.array
        Numpy array that represents the whole frame
    frame_id : int
        Current frame number.
    fps : int 
        Integer number that represents amount of frames per second.
    tr_id : int
        Ordinal number of the detected object.
    use_resize : bool
        Variable that defines if image is needed to be resized or not.
    threshold : float
        Threshold that defines if images is similiar or not.
    threshold_for_blur : int : default = 100
        Integer number that defines the score for blurry and not blurry images.
    model_name : str
        Name of the model to use. Possible options: [VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Ensemble]
    distance_metric : str
        Which distance metric to use with model. Possible options: ['cosine','euclidean','euclidean_l2']
    normalization : str
        Type of normalization to apply on input image. Possible options: ['base', 'raw', 'Facenet', 'Facenet2018','VGGFace','VGGFace2','ArcFace'].
        For more details please visit page of repository: https://github.com/serengil/deepface/blob/6c5c3511f4f569b74bfe4561b458149bbc6aa5ee/deepface/commons/functions.py#L186
    dict_with_embeddings : dict
        Dictionary that saves all embedding during the video.
    
    

    Returns:
    ---------------
    dict
        dict_with_embeddings: dict that contains embeddings of each saved photo.
    str
        tr_id: id of the folder to which detection was saved.
    """
    
    if os.path.exists(os.path.join(path_to_database,video_name)) == False:
        os.makedirs(os.path.join(path_to_database, video_name))
        path_to_database = os.path.join(path_to_database, video_name)
    else:
        path_to_database = os.path.join(path_to_database, video_name)
    
    list_of_current_folders = list(os.listdir(path_to_database))

    if list_of_current_folders == []:
        os.makedirs(os.path.join(path_to_database, str(tr_id)))
        time_of_apperance = convert_frame_to_time(frame_id, fps)
        image_to_save = frame[detect[1]:detect[3], detect[0]:detect[2], :]
        cv2.imwrite(os.path.join(path_to_database,str(tr_id), f'{tr_id}_{time_of_apperance}.jpg'), image_to_save)
        dict_with_embeddings[str(tr_id)] = [get_embedding_of_single_image(image_to_save, model_name=model_name, normalization=normalization)]

        return dict_with_embeddings, str(tr_id)
    elif list_of_current_folders != []:
        image_to_check = frame[detect[1]:detect[3], detect[0]:detect[2], :]
        if use_resize == True:
            pass
        if is_blur(img = image_to_check, threshold=threshold_for_blur) == False:
            time_of_apperance = convert_frame_to_time(frame_id,fps)
            dict_with_results_of_verification = {}
            input_embedding = get_embedding_of_single_image(image_to_check, normalization=normalization, model_name=model_name)
            # print(dict_with_embeddings)
            for key, value in dict_with_embeddings.items():
                sum_vec = []
                for item in value:
                    if sum_vec == []:
                        sum_vec = item
                    elif sum_vec != []:
                        # print(sum_vec.shape)
                        
                        sum_vec += item
                # print(len(value))
                sum_vec = np.array(sum_vec)/len(value)
                threshold_of_model = threshold
                if distance_metric == 'euclidean_l2':
                    distance = dst.findEuclideanDistance(dst.l2_normalize(sum_vec), dst.l2_normalize(input_embedding))
                elif distance_metric == 'cosine':
                    distance = dst.findCosineDistance(sum_vec, input_embedding)
                # distance = dst.findEuclideanDistance(sum_vec, input_embedding)
                # print(threshold_of_model)
                # print(distance)
                
                if distance <= threshold_of_model:
                    res = True
                elif distance > threshold_of_model:
                    res = False
                dict_with_results_of_verification[key]= [res,distance]

                

            print(f'FUTURE FILENAME: {tr_id}_{time_of_apperance}')
            print(dict_with_results_of_verification)

            list_with_results_of_verification = []
            possible_folders_to_save = dict()

            for key, value in dict_with_results_of_verification.items():
                if value[0] == True:
                    possible_folders_to_save[key] = value
                elif value[0] == False:
                    list_with_results_of_verification.append(True)


            if sum(list_with_results_of_verification) == len(list(dict_with_results_of_verification.keys())):
                os.makedirs(os.path.join(path_to_database, str(tr_id)))
                cv2.imwrite(os.path.join(path_to_database, str(tr_id), f'{tr_id}_{time_of_apperance}.jpg'), image_to_check)
                dict_with_embeddings[str(tr_id)] = [input_embedding]
                return dict_with_embeddings, str(tr_id)

            if len(list(possible_folders_to_save.keys())) == 1:
                cv2.imwrite(os.path.join(path_to_database, str(list(possible_folders_to_save.keys())[0]), f'{tr_id}_{time_of_apperance}.jpg'), image_to_check)
                dict_with_embeddings[str(list(possible_folders_to_save.keys())[0])].append(input_embedding)
                return dict_with_embeddings, str(list(possible_folders_to_save.keys())[0])
            
            elif len(list(possible_folders_to_save.keys())) > 1:
                print(possible_folders_to_save)
                # print("Something went wrong!")
                list_of_scores = []
                keys = list(possible_folders_to_save.keys())
                for key, value in possible_folders_to_save.items():
                    # print(value)
                    list_of_scores.append(value[1])
                # print(list_of_scores)
                ind = list_of_scores.index(min(list_of_scores))
                # print('Folder_to_save:')
                # print(keys[ind])
                # print('key str')
                # print(os.path.join(path_to_database, str(keys[ind]), f'{time_of_apperance}.jpg'))
                dict_with_embeddings[str(keys[ind])].append(input_embedding)
                
                cv2.imwrite(os.path.join(path_to_database, str(keys[ind]), f'{tr_id}_{time_of_apperance}.jpg'), image_to_check)
                return dict_with_embeddings, str(list(possible_folders_to_save.keys())[0])
        else:
            return dict_with_embeddings, 'no id'


def bb_intersection_over_union(boxA, boxB):
    """
    Function that computes the IoU.
    
    Arguments:
    -------------
    boxA : list
        List with coordinates of first bbox.
    boxB : list
        List with coordinates of second bbox.

    Returns:
    -------------
    float
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def create_tracker_by_name(tracker_type:str):
    """
    Function that creates open cv tracker by its name.

    Arguments:
    ---------------
    tracker_type : str
        Name of the tracker that is needed to be created. Possible options: ['KCF','MEDIANFLOW','MOSSE','CSRT']
    Returns:
    ---------------
    tracker object
        Tracker
    """
    tracker_types = ['KCF','MEDIANFLOW','MOSSE','CSRT']

    if tracker_type == tracker_types[0]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == tracker_types[1]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == tracker_types[2]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == tracker_types[3]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        print('No such type')
    return tracker

def cosine_similiarity_for_images(img_1:numpy.array, img_2:numpy.array) -> float:
    """
    Function that shows cosine similarity between two images.

    Arguments:
    ---------------
    img_1 : numpy.array
        First image as a numpy array.
    img_2 : numpy.array
        Second image as a numpy array.
    Returns:
    ---------------
    float
        Number that shows cosine similarity between two images.
    """
    if img_1.shape != img_2.shape:
        img_2 = cv2.resize(img_2, img_1.shape[:2], interpolation=cv2.INTER_AREA)
    img_1 = img_1.flatten()
    img_2 = img_2.flatten()
    img_1 = img_1/255
    img_2 = img_2/255

    similarity = -1 * (spatial.distance.cosine(img_1, img_2) - 1)

    return similarity

        
def save_the_dict(input_dict, output_path, output_file_name):
    """
    Function that saves the dict with frames into the file, so the user will be able to restore it easily.

    Arguments
    -------------
    input_dict : dict
        Dictionary with id of folders and frame numbers.
    output_path : str
        Path to the folder where you want to save the file.
    output_file : str
        Name of the output file.

    Returns:
    ------------
    None
    """
    path = os.path.join(os.getcwd(), output_path)
    f = open(os.path.join(path,output_file_name),'w')
    f.write(str(input_dict))
    f.close

def final_split(folders_list:list, automatic_search:bool = False, path_to_database:str = 'database', filename:str = 'crypto-assets.mp4', model_name:str = 'DeepFace', distance_metric:str = 'cosine', use_default_threshold:bool = True, threshold:float = 0, normalization:str = 'base', to_del:bool=True) -> None:
    """
    Split photos in garbage folders into new folders.

    Argumets:
    --------------
    folders_list : list
        List with names of the garbage folders.
    automatic_search : bool
        Variable that enable search through every folder. Very risky to use.
    path_to_database : str
        Path to the root folder of the database.
    filename : str
        Name of the video that was processed.
    model_name : str
        Name of the model to use. Possible options: [VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Ensemble]
    distance_metric : str
        Which distance metric to use with model. Possible options: ['cosine','euclidean','euclidean_l2']
    normalization : str
        Type of normalization to apply on input image. Possible options: ['base', 'raw', 'Facenet', 'Facenet2018','VGGFace','VGGFace2','ArcFace'].
        For more details please visit page of repository: https://github.com/serengil/deepface/blob/6c5c3511f4f569b74bfe4561b458149bbc6aa5ee/deepface/commons/functions.py#L186
    use_default_threshold : bool
        To use default threshold which was defined by DeepFace or not. 
    threshold : float
        Threshold that defines if images is similiar or not.
    to_del : bool
        To delete the garbage folder or not.
    
    Returns:
    -------------
    """
    current_folder = os.getcwd()
    automatic_search = False
    if automatic_search == False:
        input_folders = folders_list
    elif automatic_search == True:
        input_folders  = list(os.listdir(os.path.join(current_folder, path_to_database)))
    general_dict_with_results = {}
    path_to_database_video_folder = os.path.join(current_folder, path_to_database, filename)
    for folder in input_folders:
        dict_with_results = dict()
        current_path = os.path.join(current_folder, path_to_database, filename, folder)
        print(current_path)
        for pht_1 in tqdm(list(os.listdir(current_path))):
            print(os.path.join(current_path, pht_1))
            photo_1 = cv2.imread(os.path.join(current_path, pht_1))
            for pht_2 in tqdm(list(os.listdir(current_path))):
                photo_2 = cv2.imread(os.path.join(current_path, pht_2))
                result = reindetificate(photo_1,photo_2, threshold=threshold, model_name=model_name, distance_metric=distance_metric, normalization=normalization, use_default_threshold=use_default_threshold)
                if result == True:
                    if pht_1 not in list(dict_with_results.keys()):
                        dict_with_results[pht_1] = [pht_2]
                    elif pht_1 in list(dict_with_results.keys()):
                        dict_with_results[pht_1].append(pht_2)
        
        values = list(dict_with_results.values())
        list_with_grouped_elements = []
        for elem in values:
            if elem not in list_with_grouped_elements and len(elem)!=1:
                list_with_grouped_elements.append(elem)
        print(values)
        print(list_with_grouped_elements)
        for ind, elem in enumerate(list_with_grouped_elements):
            os.makedirs(os.path.join(path_to_database_video_folder,folder+f'.{ind+1}'))
            for item in elem:
                shutil.copy(os.path.join(current_path,item), os.path.join(path_to_database_video_folder,folder+f'.{ind+1}',item))
        if to_del == True:
            shutil.rmtree(os.path.join(current_folder, path_to_database, filename, folder))   

def open_the_file_with_dict(input_file_path):
    """
    Function that restore dictionary from file.

    Arguments:
    -----------
    input_file_path : str
        Path to the file with saved dict.

    Returns:
    -----------
    None
    """
    path = os.path.join(os.getcwd(),input_file_path)
    f = open(path,'r')
    dict_from_file = f.read()
    output_dict = ast.literal_eval(dict_from_file)
    return output_dict
  


def convert_list_to_num(img_name:str)->int:
    """
    A function that retrieves amount of seconds from name of the image. 

    Arguments:
    -------------
    img_name : str
        Name of the image that is needed to be converted.
    Returns:
    -------------
    int 
        Amount of seconds that was retrieved from the image name.
    """
    splt_name = img_name.split('-')

    num = 0
    if splt_name[0] != '00':
        num+=int(splt_name[0])*60
    num+=int(splt_name[1])
    return num


def convert_number_to_time(num:str) -> str:
    """
    A simple function that converts num into time format.

    Arguments:
    ---------------
    num : str : int
        The number that is needed to be converted.
    
    Returns:
    ---------------
    str
        Converted num into time format.
    """
    if len(str(num)) == 1:
        return f'00:0{num}'
    elif len(str(num)) == 2:
        return f'00:{num}'
    elif len(str(num))>=3:
        timestamp = f'{str(num)[:-2]}:{str(num)[-2:]}'
        return timestamp


def timelog_from_dict(input_dict, output_file_name:str = 'timelog.txt', filename:str = 'Wedding3.mp4', path_to_database:str = 'database', fps = 29, input_ids = [], garbage_folder_id = '15'):
    """
    Function that creates the timelog from dictionary with frames.

    Arguments:
    --------------
    input_dict : dict
        Dictionary where keys are ids of objects, and values are lists with frame number where corresponding objects were spotted.
    output_file_name : str
        Name of the txt file in which timelog will be saved.
    filename : str
        Name of the video for which you want to create the timelog.
    path_to_database : str
        Name of the root folder of the database.
    fps : int
        Number if frames per second.
    input_ids : list
        List of specific indexes for which you want to create the timelog.
    garbage_folder_id : str
        Id of folder to skip.   

    Returns:
    --------------
    None
    """
    path_to_timelog = os.path.join(os.getcwd(), path_to_database, filename)
    file = open(os.path.join(path_to_timelog, output_file_name),'w')
    json_file = open(os.path.join(path_to_timelog, 'timelog.json'),'w')
    dict_to_print_in_json = {}
    for key, value in input_dict.items():
        if key == 'no id' or key == str(garbage_folder_id):
            continue
        if input_ids != [] and key not in input_ids:
            continue
        grouped_files = []
        subgroup_files = []
        dict_to_print_in_json[f'Person {key}'] = {}
        dict_to_print_in_json[f'Person {key}']['Total_frames_of_presence'] = [len(value)]
        dict_to_print_in_json[f'Person {key}']['Timestamps'] = []
        string_to_write = f'Person with id: {key} were present for {len(value)} frames in total. Time stamps:  '
        for ind in range(len(value)-1):
            if (value[ind+1]) - (value[ind]) <5*fps:
                subgroup_files.append(value[ind])
            elif (value[ind+1]) - (value[ind])>=5*fps:
                subgroup_files.append(value[ind])
                grouped_files.append(subgroup_files)
                subgroup_files = []
            if ind == len(value)-2:
                subgroup_files.append(value[ind+1])
                grouped_files.append(subgroup_files)
        for item in grouped_files:
            first_i = convert_frame_to_time(item[0], fps, format_of_out= 'colon')
            second_i = convert_frame_to_time(item[-1], fps, format_of_out =  'colon')
            string_to_write+=f'{first_i} ({item[0]} frame) to {second_i} ({item[-1]} frame) \n'
            dict_to_print_in_json[f'Person {key}']['Timestamps'].append(f'{item[0]} : {item[-1]}')
        
        dict_to_print_in_json[f'Person {key}']['Total number of encounter'] = [len(grouped_files)]
        string_to_write+=f'Total number of encounter: {len(grouped_files)} \n'
        # print(string_to_write)
        file.write(string_to_write)
    json.dump(dict_to_print_in_json, json_file)
 
# with open("sample.json", "w") as outfile:
#     json.dump(dictionary, outfile)
#         



def create_time_log(path_to_database:str = 'database', filename:str = 'crypto-assets.mp4', name_of_the_output_file:str = 'timelog.txt', full_size:bool = False, input_ids= ['1']) -> None:
    """
    A function that creates a timelog after processing the video.

    Arguments:
    -------------------
    path_to_database : str
        Path to the root folder of the database.
    filename : str
        Name of the video that was processed.
    name_of_the_output_file : str
        Name of the time log file that will be created.

    Returns:
    --------------------
    None
    """
    current_folder = os.getcwd()
    path_to_use = os.path.join(current_folder, path_to_database, filename)
    list_of_folders = list(os.listdir(path_to_use))
    list_of_folders = sorted(list(map(float,list_of_folders)))
    
    if full_size == False:
        name_of_the_output_file = name_of_the_output_file[:-4]+'_of_'
        for id in input_ids:
            name_of_the_output_file+=f'{id}_'
        name_of_the_output_file+='indexes.txt'
    file = open(os.path.join(path_to_use, name_of_the_output_file),'w')
    if full_size == True:
        for folder in list_of_folders:
            if str(folder)[-1] == '0':
                folder = str(int(folder))
            elif str(folder)[-1] != '0':
                folder = str(folder)
            list_of_files = list(sorted(os.listdir(os.path.join(current_folder, path_to_database, filename, folder))))
            # print(list_of_files)
            grouped_files = []
            subgroup_files = []
            if len(list_of_files) >= 2:
                for ind in range(len(list_of_files)-1):
                    if convert_list_to_num(list_of_files[ind+1]) - convert_list_to_num(list_of_files[ind]) <2:
                        subgroup_files.append(convert_list_to_num(list_of_files[ind]))
                    elif convert_list_to_num(list_of_files[ind+1]) - convert_list_to_num(list_of_files[ind])>=2:
                        subgroup_files.append(convert_list_to_num(list_of_files[ind]))
                        grouped_files.append(subgroup_files)
                        subgroup_files = []
                    if ind == len(list_of_files)-2:
                        subgroup_files.append(convert_list_to_num(list_of_files[ind+1]))
                        grouped_files.append(subgroup_files)
                # print(grouped_files)
                string_to_print = f'Person with id: {folder} was present for '
                duration = 0
                list_with_time_stamps = []
                for item in grouped_files:
                    duration += (item[-1]) - (item[0]-1)
                    list_with_time_stamps.append(f'{item[0]} - {item[-1]}')
                string_to_print += f'{duration} seconds.  Timestamps:'
                # print(list_with_time_stamps)
                for ind, item in enumerate(list_with_time_stamps):
                    if ind+1 == 5:
                        string_to_print+='\n'
                    splt_item = item.split(' ')
                    string_to_print+=f' {convert_number_to_time(splt_item[0])} - {convert_number_to_time(splt_item[-1])}||'
                string_to_print+= f' Total encounter of person on the video: {len(list_with_time_stamps)}'
                string_to_print+='\n'
                file.write(string_to_print)
                # print(string_to_print)
    else:
        for id in input_ids:
            list_of_files = list(sorted(os.listdir(os.path.join(current_folder, path_to_database, filename, id))))
            # print(list_of_files)
            grouped_files = []
            subgroup_files = []
            if len(list_of_files) >= 2:
                for ind in range(len(list_of_files)-1):
                    if convert_list_to_num(list_of_files[ind+1]) - convert_list_to_num(list_of_files[ind]) <5:
                        subgroup_files.append(convert_list_to_num(list_of_files[ind]))
                    elif convert_list_to_num(list_of_files[ind+1]) - convert_list_to_num(list_of_files[ind])>=5:
                        subgroup_files.append(convert_list_to_num(list_of_files[ind]))
                        grouped_files.append(subgroup_files)
                        subgroup_files = []
                    if ind == len(list_of_files)-2:
                        subgroup_files.append(convert_list_to_num(list_of_files[ind+1]))
                        grouped_files.append(subgroup_files)
                # print(grouped_files)
                string_to_print = f'Person with id: {id} was present for '
                duration = 0
                list_with_time_stamps = []
                for item in grouped_files:
                    duration += (item[-1]) - (item[0]-1)
                    list_with_time_stamps.append(f'{item[0]} - {item[-1]}')
                string_to_print += f'{duration} seconds. Timestamps:'
                print(list_with_time_stamps)
                for ind, item in enumerate(list_with_time_stamps):
                    if ind+1 == 5:
                        string_to_print+='\n'
                    splt_item = item.split(' ')
                    string_to_print+=f' {convert_number_to_time(splt_item[0])} - {convert_number_to_time(splt_item[-1])}||'
                string_to_print+= f' Total encounter of person on the video: {len(list_with_time_stamps)}'
                
                file.write(string_to_print)
                print(string_to_print)



def get_embedding_of_single_image(input_img:numpy.array, model_name:str = 'ArcFace', normalization:str = 'base'):
    """
    Function that computes the embedding of a single image.

    Arguments
    ---------------
    input_img : numpy.array
        Image as numpy array.
    model_name : str
        Name of the model, that will be used to compute embedding.
    normalization : str
        Type of normalization to apply on input image. Possible options: ['base', 'raw', 'Facenet', 'Facenet2018','VGGFace','VGGFace2','ArcFace'].
        For more details please visit page of repository: https://github.com/serengil/deepface/blob/6c5c3511f4f569b74bfe4561b458149bbc6aa5ee/deepface/commons/functions.py#L186

    Returns
    ---------
    numpy.array
        Array with embedding.
    """
    print(normalization)
    print(model_name)
    assert normalization in ['base', 'raw', 'Facenet', 'Facenet2018','VGGFace','VGGFace2','ArcFace']
    input_vec = np.array(DeepFace.represent(input_img, detector_backend='skip', model_name=model_name,normalization=normalization))
    return input_vec


def define_garbage_folder(path_to_database:str, filename:str, folder:str):
    """
    Function that automaticly defines the garbage folder using cosine similarity.

    Arguments:
    -----------------

    Returns:
    -----------
    list 
        List with two elements, name of the folder, and result of check.
    """
    path_to_folder = os.getcwd()    
    cur_path =  os.path.join(path_to_folder, path_to_database,filename,folder)
    list_of_files = list(os.listdir(cur_path))
    list_of_scores = []
    for i in list_of_files:
        img_1 = cv2.imread(os.path.join(cur_path,i))
        for j in list_of_files:
            if i == j:
                continue
            img_2 = cv2.imread(os.path.join(cur_path,j))
            score = cosine_similiarity_for_images(img_1,img_2)
            list_of_scores.append(score)
    res = np.sum(np.array(list_of_scores))/len(list_of_scores)
    print(f'{folder}:{res}')
    if res < 0.88:
        return [folder, True]
    else:
        return [folder, False]
                
def save_intermediate_results_to_database(frame, detect, id, filename, path_to_database, frame_id, fps):
    """
    Function that saves an image from tracker skipping reidentication part.
    TO DO: Add the model inference because ctracker often continues to track even if there is no face.

    Arguments:
    --------------
    frame : numpy.array
        Input array that represents the image.
    detect : list
        List with lists that represents detect in next format [bb_left_x, bb_left_y, bb_right_x, bb_right_y]
    id : int
        Number that reresents id of the folder where this face were saved during detect-recogntion part.
    filename : str
        Name of the video that is currently in process.
    path_to_database:
        Path to the root folder of database.
    frame_id : int
        Current frame number.
    fps : int
        Integer number that represents the amount of frames in a second of a video.
    
    Returns:
    ------------
    None


    Returns:
    ---------------

    """
    curr_folder = os.getcwd()
    current_path = os.path.join(curr_folder, path_to_database,filename,id)
    img_to_save = frame[detect[1]:detect[3], detect[0]:detect[2], :]
    time_stamp = convert_frame_to_time(frame_id, fps)
    
    cv2.imwrite(os.path.join(path_to_database,filename,str(id), f'{time_stamp}.jpg'), img_to_save)


def scan_through_database(path_to_database, model_name, normalization):
    dict_with_embedings = {}
    list_of_files = list(os.listdir(os.path.join(path_to_database)))
    for folder in list_of_files:
        dict_with_embedings[folder]= []
        list_of_photos = list(os.listdir(os.path.join(path_to_database, folder)))
        for photo in list_of_photos:
            img_to_process = cv2.imread(os.path.join(path_to_database, folder, photo))
            out_vec = get_embedding_of_single_image(img_to_process, model_name=model_name, normalization=normalization)
            if dict_with_embedings[folder] == []:
                dict_with_embedings[folder] = out_vec
            elif dict_with_embedings[folder] != []:
                dict_with_embedings[folder] += out_vec
        dict_with_embedings[folder] = dict_with_embedings[folder]/len(list_of_photos)

    return dict_with_embedings

def preprocess_img(img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = cv2.resize(img, [112,112], interpolation=cv2.INTER_AREA)
    preprocessed_img = np.transpose(img, (2,0,1))
    preprocessed_img = np.expand_dims(preprocessed_img,axis=0)
    tensor = torch.Tensor(preprocessed_img)
    tensor = tensor.to(device=device, dtype=torch.float32)
    tensor = tensor/255
    return tensor

def load_predictor(path_config = "gpu.config.yml"):
    cfg = OmegaConf.load(path_config)
    analyzer = FaceAnalyzer(cfg.analyzer)
    predictor = analyzer.predictors['verify']
    return predictor

def inference_of_predictor(tensor, predictor):
    out_emb = predictor.run(tensor)[0].logits
    return out_emb



def send_to_database_with_tracker( dict_with_detections, path_to_database, video_name, frame, frame_id, fps, dict_with_embeddings = {}, ):
    if os.path.exists(os.path.join(path_to_database,video_name)) == False:
        os.makedirs(os.path.join(path_to_database, video_name))
        path_to_database = os.path.join(path_to_database, video_name)
    else:
        path_to_database = os.path.join(path_to_database, video_name)
    list_of_files = list(os.listdir(path_to_database))
    if len(list_of_files) == 0:
        for key_, value in dict_with_detections.items():
            os.makedirs(os.path.join(path_to_database, str(key_)))
            img_to_save = frame[value[1]:value[3], value[0]:value[2],:]
           
            time_of_apperance = convert_frame_to_time(frame_id, fps)
            cv2.imwrite(os.path.join(path_to_database,str(key_), f'{key_}_{time_of_apperance}.jpg'), img_to_save)
        return dict_with_detections, dict_with_embeddings
    elif len(list_of_files) != 0:
        updated_dict = {}
        for id_of_obj, value in dict_with_detections.items():
            if str(id_of_obj) in list_of_files:
                updated_dict[id_of_obj] = value
                img_to_save = frame[value[1]:value[3], value[0]:value[2],:]
                
                time_of_apperance = convert_frame_to_time(frame_id, fps)
                cv2.imwrite(os.path.join(path_to_database,str(id_of_obj), f'{id_of_obj}_{time_of_apperance}.jpg'), img_to_save)
            elif str(id_of_obj) not in list_of_files:
                

                img_to_check = frame[value[1]:value[3], value[0]:value[2],:]
                print(f'Print if image is blurry {is_blur(img=img_to_check, threshold=60)}')
               
                time_of_apperance = convert_frame_to_time(frame_id, fps)
                
                updated_dict[int(id_of_obj)] = value
                    # print(f'FUTURE FILENAME: {id_of_obj}_{time_of_apperance}')
                os.makedirs(os.path.join(path_to_database, str(id_of_obj)))
                    # dict_with_embeddings[str(id_of_obj)] = [out_emb_to_check]
                cv2.imwrite(os.path.join(path_to_database, str(id_of_obj), f'{id_of_obj}_{time_of_apperance}.jpg'), img_to_check)
        return updated_dict, dict_with_embeddings

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def gather_embeddings_for_database(path_to_database, filename, config_for_model = "gpu.config.yml"):
    path_img_output="/test_output.jpg"
    dict_with_embeddings = {}
    cfg = OmegaConf.load(config_for_model)
    analyzer = FaceAnalyzer(cfg.analyzer) 
    list_of_folders = list(os.listdir(os.path.join(path_to_database, filename)))
    counter = 0
    for folder in tqdm(list_of_folders):
        counter +=1
        # print(f'{counter}/{len(list_of_folders)} were processed')
        list_of_files = list(os.listdir(os.path.join(path_to_database, filename, folder)))
        dict_with_embeddings[folder] = []
        for file in list_of_files:
            path_to_the_photo = os.path.join(path_to_database, filename, folder, file)
            responce  = analyzer.run(path_image=path_to_the_photo, batch_size=cfg.batch_size, fix_img_size=cfg.fix_img_size, return_img_data=True, include_tensors=True, path_output=path_img_output)
            try:
                dict_with_embeddings[folder].append(responce.faces[0].preds['verify'].logits)
            except IndexError:
                continue
        # print(len(list_of_files))
        # print(len(dict_with_embeddings[folder]))
    return dict_with_embeddings


def merge_timelog(dict_with_frames, dict_with_merged_entities):
    merged_dict = {}
    for key, value in dict_with_merged_entities.items():
        
        merged_time_log = [] 
        for i in value:
            merged_time_log+=dict_with_frames[int(i)]
        merged_dict[key] = merged_time_log
    return merged_dict

def compare_results(dict_with_embeddings):
    values = list(dict_with_embeddings.values())
    keys = list(dict_with_embeddings.keys())
    dict_with_results = {}
    for ind in tqdm(range(len(values))):
        for j in tqdm(range(ind,len(values))):
            
            if ind == j:
                continue
            first_list = values[ind]
            second_list = values[j]
            if len(first_list) == 0 or len(second_list) == 0:
                continue
            counter = 0
            for item_1 in first_list:
                for item_2 in second_list:
                    score = cosine_similarity(item_1, item_2, dim=0)
                    if score > 0.4:
                        counter+=1
            result = counter/(len(first_list)*len(second_list))
            # print(result)
            dict_with_results[f'{keys[ind]} {keys[j]}'] = result
    return dict_with_results    

def merge_sublists(sublists):
    sublists = [sorted(list(map(int, sublist))) for sublist in sublists]
    
    result = []
    while sublists:
        curr = sublists.pop()
        overlap = [sub for sub in sublists if set(curr) & set(sub)]
        for sub in overlap:
            sublists.remove(sub)
            curr = list(set(curr) | set(sub))
        result.append(sorted(curr))
    
    merged = {}
    for sublist in result:
        key = tuple(sublist)
        if key not in merged:
            merged[key] = sublist
    return list(merged.values())

def preprocess_keys(list_with_keys):
    list_of_prop_keys = []
    for key in list_with_keys:
        key_list = key.split(' ')
        for item in key_list:
            list_of_prop_keys.append(item)
    list_of_prop_keys = list(set(list_of_prop_keys))
    return list_of_prop_keys

def find_the_single_folders(dict_with_results, path_to_database, path_to_filename):
    list_of_files = list(os.listdir(os.path.join(path_to_database, path_to_filename)))
    dict_with_results_of_check = {}
    keys_of_dict = preprocess_keys(list(dict_with_results.keys()))
    for folder in list_of_files:
        print(folder)
        counter_of_pairs_with_zeros = 0
        counter_of_all_pairs = 0
        if folder not in keys_of_dict:
            continue
        
        for key, value in dict_with_results.items():
            key_transformed = key.split(' ')
            if intersection([folder], key_transformed)!= [] and value == 0.0:
                # print(folder, key_transformed)
                counter_of_pairs_with_zeros +=1
                counter_of_all_pairs+=1
            elif intersection([folder], key_transformed)!= [] and value > 0.0:
                counter_of_all_pairs +=1
        print(folder,counter_of_pairs_with_zeros/counter_of_all_pairs )
        dict_with_results_of_check[folder] = counter_of_pairs_with_zeros/counter_of_all_pairs

    print(dict_with_results_of_check)

    list_with_folder_with_no_match = []

    for key, value in dict_with_results_of_check.items():
        if value == 1:
            list_with_folder_with_no_match.append(key)

    print(list_with_folder_with_no_match)
    return list_with_folder_with_no_match

def count_intersections(list_with_merged_folders):
    counter = 0
    for i in range(len(list_with_merged_folders)):
        for j in range(i, len(list_with_merged_folders)):
            if i == j:
                continue
            if intersection(list_with_merged_folders[i], list_with_merged_folders[j]) != []:
                counter +=1
    return counter

def merge_folders(path_to_database, file_name, dict_with_results):
    new_file_name = 'MergedFaces_DB' + file_name
    list_of_folders_to_merge = []
    counter = 0
    for key, value in dict_with_results.items():
        if value > 0.6:
            key_list = key.split(' ')
            list_of_folders_to_merge.append(key_list)
    list_of_folders_to_merge = merge_sublists(list_of_folders_to_merge)
    
    list_with_single_folders = find_the_single_folders(dict_with_results=dict_with_results,path_to_database=path_to_database, path_to_filename=file_name)
    for folder in list_with_single_folders:
        list_of_folders_to_merge.append([folder])
    while True:
        counter = count_intersections(list_with_merged_folders=list_of_folders_to_merge)
        if counter == 0:
            break
        elif counter > 0:
            list_of_folders_to_merge = merge_sublists(list_of_folders_to_merge)
    print(list_of_folders_to_merge)
    dict_with_merged_entities = {}
    for pair in list_of_folders_to_merge:
        counter+=1
        dict_with_merged_entities[str(counter)] = pair
        os.makedirs(os.path.join(path_to_database, new_file_name, str(counter)))
        for elem in pair:
            list_of_files = list(os.listdir(os.path.join(path_to_database, file_name, str(elem))))
            for file in list_of_files:
                shutil.copy(os.path.join(path_to_database, file_name, str(elem), file), os.path.join(path_to_database, new_file_name, str(counter), file ))
    return dict_with_merged_entities
                




if __name__ == '__main__':
    
    pass