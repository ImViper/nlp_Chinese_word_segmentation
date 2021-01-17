# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default='test_data/shanxi/unigram_seg.txt')
    parser.add_argument('--gold_path', default='test_data/shanxi/gold.txt')
    parser.add_argument('--pred_encoding', default='gb18030')
    parser.add_argument('--gold_encoding', default='utf-16')
    args = parser.parse_args()
    print(args.pred_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
