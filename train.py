def per_epoch_activity(num_epochs, train_iter, model, multibox_target, calc_loss_fn, optimizer,
                       accumulator, device, summary_writer, cls_eval, bbox_eval):
    for epoch in range(num_epochs):
        print("EPOCH: ", epoch)
        metric = accumulator(4)
        model.train()
        for features, labels in train_iter:
            optimizer.zero_grad()
            X, Y = features.to(device), labels.to(device)
            anchors, cls_preds, bbox_preds = model(X)

            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            loss = calc_loss_fn(cls_preds, cls_labels, bbox_preds, bbox_labels,
                                bbox_masks)
            loss.mean().backward()
            optimizer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                       bbox_labels.numel())

        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        summary_writer.add_scalar("ClassificationError/BoundingBoxError", {"ClassificationError": cls_err,
                                                                           "BoundingBoxError": bbox_mae}, epoch + 1)
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')

