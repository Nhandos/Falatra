import argparse
import cv2
import os

parser = argparse.ArgumentParser(description='Extract frames from video')
parser.add_argument('video', type=str, help='videofile')
parser.add_argument('outdir', type=str, help='output directory')
parser.add_argument('--prefix', type=str, help='prefix', default='frame_')
parser.add_argument('--format', type=str,help='image format', default='png')
parser.add_argument('--verbose', '-v', action='store_true', default=False,
    help='verbose logging')


def main(args):

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print('fatal - could not open video file')
        exit(-1)

    n_frame = 0
    while True:

        ret, frame = capture.read()
        if not ret:
            print('finished reading all frames')
            break
        n_frame +=1

        
        imgname = f'{args.prefix}{n_frame}.{args.format}'
        path = os.path.join(args.outdir, imgname)
        cv2.imwrite(path, frame)
        if args.verbose:
            print('Saved frame %d to %s' % (n_frame, path))

    capture.release()


if __name__ == '__main__':
    main(parser.parse_args())
