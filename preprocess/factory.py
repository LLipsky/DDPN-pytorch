from config.base_config import cfg
# from preprocess.refcoco import Refcoco

from preprocess.refcoco import CocoDataset

datasets={
        'flickr30k': 'Flickr30k',
        'referit': 'Referit',
        'refcoco': 'Refcoco',
        'refcoco+': 'Refcoco',
}

splits={
        'flickr30k': ['train', 'val', 'test'],
        'referit': ['train', 'val', 'test'],
        'refcoco': ['train', 'val', 'testA', 'testB', 'test'],
        'refcoco+': ['train', 'val', 'testA', 'testB', 'test']
}

def process_dataset(imdb_name):
    if imdb_name not in datasets:
        raise KeyError('Unknown dataset: {}'.format(imdb_name))
    imdb=eval(datasets[imdb_name])()

    for split in splits[imdb_name]:
        print('processing %s...',split)
        imdb.format_data(split)

if __name__=='__main__':
    dict_dir=cfg.DICT_DIR
    # refcoco=Refcoco()
    refcoco = CocoDataset()
    for split in ['train', 'val', 'testA', 'testB', 'test']:
        print('processing %s...',split)
        refcoco.format_data(split)