import argparse
import os 
import tarfile
import torch

def start_compression(path):
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path, file)
        print(f"Compressing {filepath}")

        tarfilepath = os.path.join(path, f'{os.path.basename(filepath)}.tar.gz')
        with tarfile.open(tarfilepath, 'w:gz') as tar:
            tar.add(filepath, arcname=os.path.basename(filepath))

        os.remove(filepath)


def split_tensors_into_single_chunks(path):
    save_dir = 'output'
    files = sorted(os.listdir(path))
    count = 0
    for file in files:
        filepath = os.path.join(path, file)
        print(f"Uncompressing {filepath}")

        episodes = torch.load(filepath)
        for episode in episodes:
            torch.save(episode, os.path.join(path, save_dir, f'data.{count}.pth'))
            count += 1

    print("done!")
            

def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    parser.add_argument('path', type=str, help="Path to the configuration file")
    args = parser.parse_args()
    path = args.path

    start_compression(path)

    
if __name__ == "__main__":
    main()
 
