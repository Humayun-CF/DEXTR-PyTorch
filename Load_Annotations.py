import os
import numpy as np
import nrrd
import cv2
import glob
import pydicom
from skimage import img_as_float
from skimage import img_as_ubyte

def Generate_Mask(src_path):
    file = open(os.path.join(src_path, 'trainval.txt'), 'w')
    file_tr = open(os.path.join(src_path, 'train.txt'), 'w')
    file_val= open(os.path.join(src_path, 'val.txt'), 'w')
    dst_path = src_path + '_Segmentation'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    for filename in glob.glob(os.path.join(src_path,'*.nrrd')):
        # filename = 'ProstateDx-01-0001.nrrd'
        file_name = os.path.basename(filename)
        file_name , file_ext = os.path.splitext(file_name)
        print(filename)

        image_data, options = nrrd.read(os.path.join(src_path, file_name+file_ext))
        z_stacks = image_data.shape[2]

        for i in range(z_stacks):
            if len(np.unique(image_data[:,:,i])) > 1:
                mask = image_data[:,:,i]
                cv2.imwrite(os.path.join(dst_path, file_name+'___'+str(i)+'.png'), mask)

                # mask = cv2.inRange(image_data[:,:,i], 0.66, 2.)
                # cv2.imwrite(os.path.join(dst_path, file_name+'___'+str(i)+'.png'), mask)
                # mask = cv2.inRange(image_data[:,:,i], 0.66, 1.33)
                # cv2.imwrite(os.path.join(dst_path, file_name+'___'+str(i)+'_A.png'), mask)
                # mask = cv2.inRange(image_data[:,:,i], 1.5, 2.)
                # cv2.imwrite(os.path.join(dst_path, file_name+'___'+str(i)+'_B.png'), mask)
                file.write(file_name+'___'+str(i) + '\n')
                if i%4 == 0:
                    file_val.write(file_name+'___'+str(i) + '\n')
                else:
                    file_tr.write(file_name+'___'+str(i) + '\n')
    file.close()
    file_tr.close()
    file_val.close()


def Generate_GrayScale(src_path):
    file = open(os.path.join(src_path, 'trainval2.txt'), 'w')
    dst_path = src_path + '_JPG'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    # 1. First get the names of all folders
    for root, dirs, files in os.walk(src_path):
        Study_IDs = dirs
        break

    for Study_ID in Study_IDs:
        for root, dirs, files in os.walk(os.path.join(src_path, Study_ID)):
            if len(files) > 0:
                for i, filename in enumerate(files):
                    # print(os.path.join(root,filename))
                    dcm = pydicom.read_file(os.path.join(root, filename))

                    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
                    ConstPixelDims = (int(dcm.Rows), int(dcm.Columns)) #, len(files))

                    # Load spacing values (in mm)
                    ConstPixelSpacing = (float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1])) #, float(dcm.SliceThickness))

                    x = np.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0])
                    y = np.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])
                    # z = np.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])

                    ArrayDicom = np.zeros(ConstPixelDims, dtype=dcm.pixel_array.dtype)
                    ArrayDicom[:, :] = dcm.pixel_array
                    cv2.imwrite(os.path.join(dst_path, Study_ID + '___' + str(i) + '.png'), ArrayDicom)

                    file.write(Study_ID + '___' + str(i) + '\n')
    file.close()


src_path = '/data/Humayun/Arterys/Dataset_NCI_ISBI_Prostate_Challenge/NCI_ISBI_Challenge-Prostate3T'
Generate_Mask(src_path)

src_path = '/data/Humayun/Arterys/Dataset_NCI_ISBI_Prostate_Challenge/Prostate-3T'
Generate_GrayScale(src_path)
