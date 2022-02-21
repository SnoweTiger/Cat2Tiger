import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as transforms


from model import Generator


def save_checkpoint(epoch, G_AB, G_BA, D_A, D_B, WEIGHT_PATH):
  torch.save(G_AB.state_dict(), f'{WEIGHT_PATH}/{epoch}_G_AB.pth')
  torch.save(G_BA.state_dict(), f'{WEIGHT_PATH}/{epoch}_G_BA.pth')
  torch.save(D_A.state_dict(), f'{WEIGHT_PATH}/{epoch}_D_A.pth')
  torch.save(D_B.state_dict(), f'{WEIGHT_PATH}/{epoch}_D_B.pth')

def load_checkpoint(epoch, G_AB, G_BA, D_A, D_B, WEIGHT_PATH):
  G_AB.load_state_dict(torch.load(f'{WEIGHT_PATH}/{epoch}_G_AB.pth', map_location=device))
  G_BA.load_state_dict(torch.load(f'{WEIGHT_PATH}/{epoch}_G_BA.pth', map_location=device))
  D_A.load_state_dict(torch.load(f'{WEIGHT_PATH}/{epoch}_D_A.pth', map_location=device))
  D_B.load_state_dict(torch.load(f'{WEIGHT_PATH}/{epoch}_D_B.pth', map_location=device))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)




def denorm(x, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    return transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())(x)

def allocated_GPU_memory():
    torch.cuda.empty_cache()
    return round(torch.cuda.memory_allocated() / (10 ** 6) , 2)


def flip_RGB(image):
    return image[:, :, [2, 1, 0]]

def make_GIF(images, filename, frame_duration, n_loop = 0): # n_loop = 0 for inf. loops
    images[0].save(filename, save_all=True, append_images=images[1:],
                   optimize=False, duration=frame_duration, loop=0)

def load_G_AB(epoch, G_AB, WEIGHT_PATH, device = 'cpu'):
    G_AB.load_state_dict(torch.load(f'{WEIGHT_PATH}/{epoch}_G_AB.pth', map_location=device))

def load_gen(load_epoch = 28, device = 'cpu', WEIGHT_PATH = None):
    G_AB = Generator(3, 6, 3).to(device)

    if load_epoch and WEIGHT_PATH != None:
        load_G_AB(load_epoch, G_AB, WEIGHT_PATH, device)
        print(f'Loaded weight (epoch {load_epoch})')
    else:
        G_AB.apply(weights_init_normal)
        print('Weight not found. Init normal weight.')

    return G_AB

def get_fake_image(model, image, device = 'cpu'):

    image_H, image_W, _ = image.shape
    image_net = 256
    # print(image.shape)

    if image_W >= image_H:
        video_out_H = int(image_net)
        video_out_W = int(image_net * (image_W / image_H))
    else:
        video_out_W = int(image_net)
        video_out_H = int(image_net * (image_H / image_W))

    transform_totensor = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = cv2.resize(image,(256,256),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    image = flip_RGB(image)
    image = transform_totensor(image).unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image_fake = model(image.to(device))
        image_fake = image_fake.to('cpu')

    image_fake = image_fake.squeeze()
    image_fake = denorm(image_fake)
    image_fake = transforms.Resize((video_out_H, video_out_W), transforms.functional.InterpolationMode.BICUBIC)(image_fake)
    image_fake = image_fake.permute(1,2,0)#.type(torch.int8)
    image_fake = (image_fake * 255).type(torch.uint8)
    image_fake = np.array(image_fake)

    return image_fake

def convert_image(device = 'cpu',
                filename_image = 'image.jpg',
                filename_out = 'image_fake.jpg',
                load_epoch = 28, WEIGHT_PATH = None
                ):

    image = cv2.imread(filename_image)

    G_AB = load_gen(load_epoch, device, WEIGHT_PATH)

    img_fake = get_fake_image(G_AB, image, device = 'cpu')

    # To CV2 image
    # img_fake = img_fake * 255
    img_fake = flip_RGB(img_fake) # flip RGB->BGR or BGR->RGB
    # img_fake = np.array(img_fake)
    cv2.imwrite(filename_out, img_fake)

def convert_video(device = 'cpu',
                    in_video_filename = 'video.mp4',
                    # out_video_path = 'video_fake.mp4',
                    load_epoch = 28, WEIGHT_PATH = None,
                    start_frame = 0, lengh = 0,
                    out = 'video'): # in frames, 0 for max

    # lengh = 30
    image_net = 256
    out_video_filename = in_video_filename.split('.')[0] + '_fake'

    video_in = cv2.VideoCapture(in_video_filename)
    video_in_W = int(video_in.get(3))
    video_in_H = int(video_in.get(4))
    fps = video_in.get(cv2.CAP_PROP_FPS)
    frames_count = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_in_W >= video_in_H:
        video_out_H = int(image_net)
        video_out_W = int(image_net * (video_in_W / video_in_H))
    else:
        video_out_W = int(image_net)
        video_out_H = int(image_net * (video_in_H / video_in_W))

    print(f'Video in: {video_in_W}x{video_in_H}, fps:{round(fps,2)}, frames:{frames_count}')

    if lengh == 0: lengh = frames_count - start_frame
    images = []

    print(f'Read {lengh}/{frames_count} frames from video. Start frame {start_frame}. ') #End frame {start_frame+lengh-1}')
    for frame_n in tqdm(range(frames_count)):
        if frame_n >= start_frame and frame_n < (start_frame + lengh):
            ret, frame_in = video_in.read()
            images.append(frame_in)
    video_in.release()

    print('Generate fake images')
    G_AB = load_gen(load_epoch, device, WEIGHT_PATH)
    images_fake = []
    for image in tqdm(images):
        image_fake = get_fake_image(G_AB, image, device = 'cpu')
        images_fake.append(image_fake)


    if out == 'gif':
        print('Gif out: ')
        images_fake_ = []
        frame_duration = 1 / fps * 1000
        for image_fake in tqdm(images_fake):
            image_fake = Image.fromarray(image_fake, "RGB")
            images_fake_.append(image_fake)
        make_GIF(images_fake_, out_video_filename+'.gif', frame_duration)
    else:
        fourcc = cv2.VideoWriter_fourcc('x', '2', '6', '4')
        video_out = cv2.VideoWriter(out_video_filename+'.mp4', fourcc, fps, (video_out_W, video_out_H))
        print(f'Video out: {video_out_W}x{video_out_H}, fps:{round(fps,2)}, frames:{len(images_fake)}')
        for image_fake in tqdm(images_fake):
            image_fake = flip_RGB(image_fake)
            video_out.write(image_fake)
        video_out.release()
