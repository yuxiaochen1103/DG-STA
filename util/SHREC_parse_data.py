import numpy as np
min_seq = 8

#Change the path to your downloaded SHREC2017 dataset
dataset_fold = "/gpu2/yc984/hand/dataset/SHREC17/HandGestureDataset_SHREC2017"

def split_train_test(data_cfg):
    def parse_file(data_file,data_cfg):
        #parse train / test file


        label_list = []
        all_data = []
        for line in data_file:
            data_ele = {}
            data = line.split() #【id_gesture， id_finger， id_subject， id_essai， 14_labels， 28_labels size_sequence】
            #video label
            if data_cfg == 0:
                label = int(data[4])
            elif data_cfg == 1:
                label = int(data[5])
            label_list.append(label) #add label to label list
            data_ele["label"] = label
            #video
            video = []
            joint_path = dataset_fold + "/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt".format(data[0],data[1],data[2],data[3])
            joint_file = open(joint_path)
            for joint_line in joint_file:
                joint_data = joint_line.split()
                joint_data = [float(ele) for ele in joint_data]#convert to float
                joint_data = np.array(joint_data).reshape(22,3)#[[x1,y1,z1], [x2,y2,z2],.....]
                video.append(joint_data)
            while len(video) < min_seq:
                video.append(video[-1])
            data_ele["skeleton"] = video
            data_ele["name"] = line
            all_data.append(data_ele)
            joint_file.close()
        return all_data, label_list



    print("loading training data........")
    train_path = dataset_fold + "/train_gestures.txt"
    train_file = open(train_path)
    train_data, train_label = parse_file(train_file,data_cfg)
    assert len(train_data) == len(train_label)

    print("training data num {}".format(len(train_data)))

    print("loading testing data........")
    test_path = dataset_fold + "/test_gestures.txt"
    test_file = open(test_path)
    test_data, test_label = parse_file(test_file, data_cfg)
    assert len(test_data) == len(test_label)

    print("testing data num {}".format(len(test_data)))

    return train_data, test_data