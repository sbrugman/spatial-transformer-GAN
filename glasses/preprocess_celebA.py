import numpy as np
import scipy.misc
import argparse


def parse_arguments():
	parser = argparse.ArgumentParser(description='Celeb A dataset preprocess script')
	parser.add_argument('celebA_path', type=str, default='celebA', help='The path to the Celeb A dataset. The dataset can be obtained from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.')
	parser.add_argument('--output_path', type=str, default='dataset', help='The path to store the preprocessed dataset.')
	return parser.parse_args()


def preprocess_celeb_a():
	part_dict = {}
	with open("{0}/list_eval_partition.txt".format(celebA_path)) as file:
		for line in file:
			token = line.strip().split()
			part_dict[token[0]] = token[1]

	attr_dict = {}
	with open("{0}/list_attr_celeba.txt".format(celebA_path)) as file:
		skip2rows = 0
		for line in file:
			skip2rows += 1
			if skip2rows<=2: continue
			token = line.strip().split()
			attr_dict[token[0]] = token[1:]

	for type in ["train","test"]:
		L = []
		for key in part_dict:
			if part_dict[key]==("0" if type=="train" else "2" if type=="test" else None):
				L.append(key)
		count = len(L)
		images = np.ones([count,218,178,3],dtype=np.uint8)
		attributes = np.ones([count,40],dtype=np.bool)
		for i in range(len(L)):
			key = L[i]
			img = scipy.misc.imread("{0}/img_align_celeba_png/{1}".format(celebA_path,key[:-4]+".png"))
			images[i] = img
			attr = [True if e=="1" else False for e in attr_dict[key]]
			attributes[i] = attr
			print("{0} {1}/{2} done".format(type,i,len(L)))
		np.save("{0}/image_{1}.npy".format(output_path,type),images)
		np.save("{0}/attribute_{1}.npy".format(output_path,type),attributes)

		
if __name__ == "__main__":
        args = parse_arguments()
	preprocess_celeb_a(args.celebA_path, args.output_path)


