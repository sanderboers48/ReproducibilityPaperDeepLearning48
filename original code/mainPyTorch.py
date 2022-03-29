#
# train_dataloader = DataLoader(np.hstack((X_train_5, y_train_5)), batch_size=64)
# test_dataloader = DataLoader(np.hstack((X_test_5, y_test_5)), batch_size=64)
#
#
# def train_loop(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         # Compute prediction and loss
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
# def test_loop(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0
#
#     with torch.no_grad():
#         for X, y in dataloader:
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#
#
# for t in range(EPOCHS):
#     print(f"Epoch {t + 1}\n-------------------------------")
#     train_loop(train_dataloader, lstm_model, loss_fn, optimizer)
#     test_loop(test_dataloader, lstm_model, loss_fn)
# print("Done!")
#
# print('Finished Training')

# import time
#
# def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=100, device='cpu'):
#
#     print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
#           (type(model).__name__, type(optimizer).__name__,
#            optimizer.param_groups[0]['lr'], epochs, device))
#
#     history = {}  # Collects per-epoch loss and acc like Keras' fit().
#     history['loss'] = []
#     history['val_loss'] = []
#     history['acc'] = []
#     history['val_acc'] = []
#
#     start_time_sec = time.time()
#
#     for epoch in range(1, epochs + 1):
#
#         # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
#         model.train()
#         train_loss = 0.0
#         num_train_correct = 0
#         num_train_examples = 0
#
#         for batch in train_dl:
#             optimizer.zero_grad()
#
#             x = batch[0].to(device)
#             y = batch[1].to(device)
#             yhat = model(x)
#             loss = loss_fn(yhat, y)
#
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.data.item() * x.size(0)
#             num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
#             num_train_examples += x.shape[0]
#
#         train_acc = num_train_correct / num_train_examples
#         train_loss = train_loss / len(train_dl.dataset)
#
#         # --- EVALUATE ON VALIDATION SET -------------------------------------
#         model.eval()
#         val_loss = 0.0
#         num_val_correct = 0
#         num_val_examples = 0
#
#         for batch in val_dl:
#             x = batch[0].to(device)
#             y = batch[1].to(device)
#             yhat = model(x)
#             loss = loss_fn(yhat, y)
#
#             val_loss += loss.data.item() * x.size(0)
#             num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
#             num_val_examples += y.shape[0]
#
#         val_acc = num_val_correct / num_val_examples
#         val_loss = val_loss / len(val_dl.dataset)
#
#         if epoch == 1 or epoch % 10 == 0:
#             print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
#                   (epoch, epochs, train_loss, train_acc, val_loss, val_acc))
#
#         history['loss'].append(train_loss)
#         history['val_loss'].append(val_loss)
#         history['acc'].append(train_acc)
#         history['val_acc'].append(val_acc)
#
#     # END OF TRAINING LOOP
#
#     end_time_sec = time.time()
#     total_time_sec = end_time_sec - start_time_sec
#     time_per_epoch_sec = total_time_sec / epochs
#     print()
#     print('Time total:     %5.2f sec' % (total_time_sec))
#     print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))
#
#     return history
#
# history = train(
#     model=lstm_model,
#     optimizer=optimizer,
#     loss_fn=loss_fn,
#     train_dl=train_loader,
#     val_dl=val_loader,)

batch_size = 64