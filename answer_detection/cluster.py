from sklearn.cluster import KMeans
def cluster_bboxes_by_y(bbox_list, num_clusters=25):

    # Extract the y-coordinates of center points
    y_coords = [[bbox[4][1]] for bbox in bbox_list]

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(y_coords)

    # Organize bounding boxes into clusters
    clusters = [[] for _ in range(num_clusters)]
    for bbox, label in zip(bbox_list, labels):
        clusters[label].append(bbox)
        
    clusters.sort(key=lambda cluster: cluster[0][4][1] if cluster else float('inf'))

    return clusters

def find_mean(lst):
    return sum(lst) / len(lst) if lst else 0

def get_slots(clusters):
    # Initialize lists for left and right x-coordinates of bounding boxes in each slot
    boxes_l = [[], [], [], []]
    boxes_r = [[], [], [], []]

    # Collect left and right x-coordinates for each bounding box in the clusters
    for cluster in clusters:
        for i, box in enumerate(cluster):
            boxes_l[i].append(box[2][0])  # left x-coordinate
            boxes_r[i].append(box[3][0])  # right x-coordinate

    # Calculate mean for each slot
    return tuple((find_mean(boxes_l[i]), find_mean(boxes_r[i])) for i in range(4))

def get_col(bbox, slots):
    center_x = bbox[4][0]
    if center_x>=slots[0][0] and center_x<= slots[0][1]:
        return 0
    if center_x>=slots[1][0] and center_x<= slots[1][1]:
        return 1
    if center_x>=slots[2][0] and center_x<= slots[2][1]:
        return 2
    if center_x>=slots[3][0] and center_x<= slots[3][1]:
        return 3
    return -100000000

def add_row_col(cluster, slots, row_number):
    # Iterate over each bounding box in the cluster
    for i in range(len(cluster)):
        bbox = cluster[i]
        
        col_number = get_col(bbox, slots)
        
        bbox = bbox + ((row_number, col_number),)

        cluster[i] = bbox
    return cluster

def get_question_number(clusters,total_ques,total_rows,slots):

    answer_boxes = {}
    for i in range(total_ques):
        answer_boxes[i+1] = None
    for row_number in range(len(clusters)):
        cluster = clusters[row_number]
        cluster = add_row_col(cluster,slots,row_number)
        for box in cluster:
            marked_option = box[0]
            top_left = box[2]
            bottom_right = box[3]
            question_number = box[5][0] + box[5][1]*total_rows+1

            temp ={
                "op":marked_option,
                "tl": top_left,
                "br":bottom_right,
                "qn":question_number}
            answer_boxes[question_number] = temp
    return answer_boxes
        
        
        
        