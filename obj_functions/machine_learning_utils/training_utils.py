import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import datetime
import csv
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import time
import cv2
from obj_functions.machine_learning_utils.sam.sam import SAM
from obj_functions.machine_learning_utils.pytorch_grad_cam.gradcam import GradCam, GuidedBackpropReLUModel, preprocess_image, deprocess_image, show_cam_on_image

def accuracy(y, target):
    pred = y.data.max(1, keepdim=True)[1]
    acc = pred.eq(target.data.view_as(pred)).cpu().sum()
    return acc


def start_train(model, train_data, test_data, cuda_id, save_path):
    device = torch.device("cuda", cuda_id) if torch.cuda.is_available() else torch.device("cpu")
    print_resource(torch.cuda.is_available(), cuda_id, save_path)
    model = model.to(device)
    cudnn.benchmark = True
    optimizer = optim.SGD(model.parameters(), lr=model.lr, momentum=model.momentum, weight_decay=model.weight_decay, nesterov=model.nesterov)
    loss_func = nn.CrossEntropyLoss().cuda()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=model.lr_step, gamma=model.lr_decay)

    rsl, rsl_keys = [], ["lr", "epoch", "TrA", "TrL", "TeA", "TeL", "Time"]
    loss_min, acc_max = 1.0e+8, 0.0
    print_result(rsl_keys, save_path)

    for epoch in range(model.epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_acc, train_loss = train(device, optimizer, model, train_data, loss_func)
        model.eval()
        with torch.no_grad():
            test_acc, test_loss = test(device, optimizer, model, test_data, loss_func)
        scheduler.step()

        time_now = str(datetime.datetime.today())[:-7]
        rsl.append({k: v for k, v in zip(rsl_keys, [lr, epoch + 1, train_acc, train_loss, test_acc, test_loss, time_now])})
        loss_min, acc_max = min(loss_min, test_loss), max(acc_max, test_acc)
        print_result(list(rsl[-1].values()), save_path)

    print_last_result([loss_min, acc_max], save_path)

    return loss_min, acc_max

def start_train_t(model, train_data, test_data, cuda_id, save_path, args, train_idx, valid_idx):
    device = torch.device("cuda", cuda_id) if torch.cuda.is_available() else torch.device("cpu")
    print_resource(torch.cuda.is_available(), cuda_id, save_path)
    model = model.to(device)
    cudnn.benchmark = True
    if args.opt.upper() == "SGD":
        optimizer = optim.SGD(\
                            model.parameters(), \
                            lr=args.lr, \
                            momentum=args.momentum, \
                            weight_decay=args.weight_decay, \
                            nesterov=args.nesterov \
                            )
    elif args.opt.upper() == "ADAM":
        optimizer = optim.Adam(\
                            model.parameters(), \
                            lr=args.lr, \
                            weight_decay=args.weight_decay, \
                            )
    else:
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer = SAM(\
            model.parameters(), \
            base_optimizer, \
            rho=args.rho, \
            lr=args.lr, \
            momentum=args.momentum, \
            weight_decay=args.weight_decay, \
            nesterov=args.nesterov \
            )
    loss_func = nn.CrossEntropyLoss().cuda()
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step, gamma=args.lr_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    save_model_path = "{0}".format(*save_path.split("."))

    rsl, rsl_keys = [], ["lr", "epoch", "TrA", "TrL", "TeA", "TeL", "Time"]
    loss_min, acc_max = 1.0e+8, 0.0
    print_result(rsl_keys, save_path)

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_acc, train_loss = train(device, optimizer, model, train_data, loss_func, args)
        model.eval()
        with torch.no_grad():
            test_acc, test_loss = test(device, optimizer, model, test_data, loss_func)
        scheduler.step(test_loss)

        time_now = str(datetime.datetime.today())[:-7]
        rsl.append({k: v for k, v in zip(rsl_keys, [lr, epoch + 1, train_acc, train_loss, test_acc, test_loss, time_now])})
        if acc_max <= test_acc and loss_min >= test_loss:
            path = f"{save_model_path}.pth"
            best_model = torch.save(model.state_dict(), path)

        loss_min, acc_max = min(loss_min, test_loss), max(acc_max, test_acc)
        print_result(list(rsl[-1].values()), save_path)
    

        if epoch == int(args.epochs-1) and args.gcam:
            model.load_state_dict(torch.load(path))
            with torch.no_grad():
                image_path = get_wrong_image(device, optimizer, model, test_data, loss_func, save_path, valid_idx)
            bar = tqdm(desc="GradCam", total=len(image_path), leave=False)
            for i in image_path:
                spath = "{0}_".format(*save_path.split("."))
                gc(i, spath, model, args)
                bar.update()
            bar.close()
        
    print_last_result([loss_min, acc_max], save_path)

    return loss_min, acc_max

def train(device, optimizer, model, train_data, loss_func, args):
    train_acc, train_loss, n_train = 0, 0, 0
    model.train()
    bar = tqdm(desc="Training", total=len(train_data), leave=False)
    for data, target in train_data:
        data, target = data.to(device), target.to(device)

        loss = loss_func(model(data), target)

        if args.opt.upper() != "SAM":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # fisrt forward-backward pass
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            loss_func(model(data), target).mean().backward()
            optimizer.second_step(zero_grad=True)
        
        train_acc += accuracy(model(data), target)
        train_loss += loss.item() * target.size(0)
        n_train += target.size(0)

        bar.set_description("Loss: {0:.3f}, Accuracy: {1:.3f}".format(train_loss / n_train, float(train_acc) / n_train))
        bar.update()
    bar.close()

    return float(train_acc) / n_train, train_loss / n_train


def test(device, optimizer, model, test_data, loss_func):
    test_acc, test_loss, n_test = 0, 0, 0
    bar = tqdm(desc="Testing", total=len(test_data), leave=False)

    for data, target in test_data:
        data, target = data.to(device), target.to(device)
        y = model(data)
        loss = loss_func(y, target)
        test_acc += accuracy(y, target)
        test_loss += loss.item() * target.size(0)
        n_test += target.size(0)

        bar.update()
    bar.close()

    return float(test_acc) / n_test, test_loss / n_test

def get_wrong_image(device, optimizer, model, test_data, loss_func, save_path, valid_idx):
    test_acc, test_loss, n_test = 0, 0, 0
    bar = tqdm(desc="Get Wrong Image", total=len(test_data), leave=False)
    cam_path = "./obj_functions/machine_learning_utils/datasets/train_images/train_"
    path = "{0}_".format(*save_path.split("."))
    idx = 0
    image_path = []
    for data, target in test_data:
        data, target = data.to(device), target.to(device)
        y = model(data)
        loss = loss_func(y, target)
        test_acc += accuracy(y, target)
        pred = y.data.max(1, keepdim=True)[1]
        if pred.eq(target.data.view_as(pred)).cpu().sum().item() == 0:

            vis_path = cam_path+"{0:04d}".format(valid_idx[idx])+".jpg"
            image_path.append(vis_path)

        bar.update()
        idx += 1
    bar.close()
    return image_path


def gc(vis_path, path, model, args):
    # if args.mtra.upper() == "EFF0" or args.mtra.upper() == "EFF7":
    #     feature_module=model.fc
    # else:
    feature_module=model.layer4

    grad_cam = GradCam(model=model, feature_module=feature_module, \
                    target_layer_names=["2"], use_cuda=True)
    img = cv2.imread(vis_path, 1)
    img = np.float32(img) / 255
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)

    target_category = None
    grayscale_cam = grad_cam(input_img, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    # gb = gb_model(input_img, target_category=target_category)
    # gb = gb.transpose((1, 2, 0))

    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask*gb)
    # gb = deprocess_image(gb)

    name = "{5}".format(*vis_path.split("/"))
    os.makedirs(path+"cam/", exist_ok=True)
    cv2.imwrite(path+"cam/"+name, cam)
    # cv2.imwrite(path+"gb"+name, gb)
    # cv2.imwrite(path+"cam_gb"+name, cam_gb)

def print_last_result(values, save_path):
    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar=" ")
        s = "\n MinTestLoss: {}\n MaxTestAcc: {}".format(*values)
        writer.writerow([s])


def print_result(values, save_path):
    if type(values[0]) == str:
        s = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(*values)
        print(s)
        s = "\t{}\t\t{}\t{}\t\t{}\t\t{}\t\t{}\t\t{}".format(*values)
    else:
        s = "{:.4f}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(*values)
        print(s)
        s = "\t{:.4f}\t{}\t\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(*values)

    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar=" ")
        writer.writerow([s])

    record_login(save_path)


def record_login(save_path):
    current_time = str(datetime.datetime.today())[:-7]
    current_pid = os.environ["JOB_ID"] if "JOB_ID" in os.environ.keys() else os.getpid()

    with open(save_path[:-12] + "last_login.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["{}+{}".format(current_pid, current_time)])


def print_resource(is_available, gpu_id, save_path):
    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar=" ")
        s = "Computing on GPU{}\n".format(gpu_id) if is_available else "Not Computing on GPU\n"
        print(s)
        writer.writerow([s])


def print_config(hp_dict, save_path, is_out_of_domain=False):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quotechar=" ")
        s = "### Hyperparameters ###\n"

        for name, value in hp_dict.items():
            s += "{}: {}\n".format(name, value)
        writer.writerow([s])

        if is_out_of_domain:
            s = "Out of Domain\n"
            s += "\nMinTestLoss: {}\nMaxTestAcc: {}".format(1.0e+8, 0.0)
            writer.writerow([s])

    record_login(save_path)
