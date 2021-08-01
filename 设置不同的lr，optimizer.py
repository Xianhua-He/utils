if args.no_bn_wd:
    optimizer = torch.optim.SGD(
        [{'params': bn_params, 'weight_decay': 0}, {'params': rest_params, 'weight_decay': args.weight_decay}],
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov)
else:
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)


#####################################################
modules_ori = [model.resnet_features]
modules_new = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
params_list = []
for module in modules_ori:
    params_list.append(dict(params=module.parameters(), lr=args.base_lr))
for module in modules_new:
    params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)