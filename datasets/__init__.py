from .dataset import build_multithumos, build_charades, build_thumos14, build_activitynet

def build_dataset(image_set, args):
    if args.dataset == 'multithumos':
        return build_multithumos(image_set, args)
    elif args.dataset == 'charades':
        return build_charades(image_set, args)
    elif args.dataset == 'thumos14':
        return build_thumos14(image_set, args)
    elif args.dataset == 'activitynet':
        return build_activitynet(image_set, args)
    else:
        raise ValueError(f'dataset {args.dataset} not implemented')
