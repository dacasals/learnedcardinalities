import argparse
import time
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mscn.util import *
from mscn.data_sparql import get_train_datasets, load_data, make_dataset
from mscn.model import SetConv
import pandas as pd

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, predicates, joins, predicate_uris, targets, sample_masks, predicate_masks, join_masks, predicate_uri_masks = data_batch

        if cuda:
            samples, predicates, predicates_uri, joins, targets = samples.cuda(), predicates.cuda(), predicate_uris.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        samples, predicates, predicate_uris, joins, targets = Variable(samples), Variable(predicates), Variable(predicate_uris), Variable(joins), Variable(targets)
        sample_masks, predicate_masks, predicate_uri_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(predicate_uri_masks), Variable(join_masks)

        t = time.time()
        outputs = model(samples, predicates, predicate_uris, joins, sample_masks, predicate_masks, predicate_uri_masks, join_masks)

        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


def train_and_predict(workload_name, url_queries, num_queries, num_samples, num_epochs, batch_size, hid_units, cuda, delimiter):
    """

    :param workload_name:
    :param url_queries:
    :param num_queries:
    :param num_samples:
    :param num_epochs:
    :param batch_size:
    :param hid_units:
    :param cuda:
    :param delimiter_col: Delimiter of column
    :return:
    """
    # Load training and validation data
    num_materialized_samples = num_samples
    df_file = pd.read_csv(url_queries, delimiter=delimiter, engine='python')
    msk = np.random.rand(len(df_file)) < 0.85
    train = df_file[msk]
    test = df_file[~msk]
    dicts, column_min_max_vals, column_min_max_cards, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_v1_joins, max_num_predicates, max_num_predicates_uris, train_data, test_data = get_train_datasets(
        train,
        num_queries,
        num_materialized_samples,
        delimiter=delimiter
    )
    table2vec, column2vec, op2vec, join2vec, join_v1_2vec, types_injoin_2vec, column2uris_vec, op2uris_vec, uris2uris_vec = dicts
    # Train model
    sample_feats = len(table2vec) + num_materialized_samples
    predicate_feats = len(column2vec) + len(op2vec) + 1
    predicateuri_feats = len(column2uris_vec) + len(op2uris_vec)+ len(uris2uris_vec) + 1
    join_feats = len(join_v1_2vec) * 2 + len(types_injoin_2vec)

    model = SetConv(sample_feats, predicate_feats, predicateuri_feats, join_feats, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    model.train()
    for epoch in range(num_epochs):
        loss_total = 0.

        for batch_idx, data_batch in enumerate(train_data_loader):

            samples, predicates, joins, predicate_uris, targets, sample_masks, predicate_masks, join_masks, predicate_uri_masks = data_batch

            if cuda:
                samples, predicates,predicates_uri, joins, targets = samples.cuda(), predicates.cuda(), predicate_uris.cuda(), joins.cuda(), targets.cuda()
                sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
            samples, predicates,predicate_uris, joins, targets = Variable(samples), Variable(predicates), Variable(predicate_uris), Variable(joins), Variable(targets)
            sample_masks, predicate_masks,predicate_uri_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(predicate_uri_masks), Variable(join_masks)

            optimizer.zero_grad()
            outputs = model(samples, predicates, predicate_uris, joins, sample_masks, predicate_masks, predicate_uri_masks, join_masks)
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))

    # Get final training and validation set predictions
    preds_train, t_total = predict(model, train_data_loader, cuda)
    print("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per validation sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error training set:")
    print_qerror(preds_train_unnorm, labels_train_unnorm)

    print("\nQ-Error validation set:")
    print_qerror(preds_test_unnorm, labels_test_unnorm)
    print("")

    # Load test data
    # file_name = "workloads/" + workload_name
    joins, joins_v1, predicates, predicates_uris, tables, samples, label = load_data(test, num_materialized_samples)
    # Get feature encoding and proper normalization
    # samples_test = encode_tables(tables, table2vec)
    # predicates_test, joins_test = encode_sparql_data(predicates, joins, column2vec, op2vec, join2vec)
    # predicates_test, joins_test = encode_sparql_data(column_min_max_vals, column_min_max_cards, predicates, predicates_uris, joins, column2vec,
    #                    op2vec, join2vec, column2uris_vec, op2uris_vec, uris2uris_vec)
    samples_enc, \
    joins_v1_enc, \
    predicates_enc, \
    joins_enc, \
    predicates_uris_enc = encode_sparql_data(
        column_min_max_vals,
        column_min_max_cards,
        tables,
        table2vec,
        joins_v1,
        join_v1_2vec,
        types_injoin_2vec,
        predicates,
        predicates_uris,
        joins,
        column2vec,
        op2vec,
        join2vec,
        column2uris_vec,
        op2uris_vec,
        uris2uris_vec
    )
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    print("Number of test samples: {}".format(len(labels_test)))

    max_num_predicates = max([len(p) for p in predicates_enc])
    max_num_joins = max([len(j) for j in joins_enc])
    max_num_v1_joins = max([len(j) for j in joins_v1_enc])
    max_num_predicates_uris = max([len(j) for j in predicates_uris_enc])

    ##################


    # Get test set predictions
    test_data = [samples_enc, predicates_enc, joins_enc, joins_v1_enc, predicates_uris_enc]
    test_dataset = make_dataset(
        *test_data,
        labels=labels_test,
        max_num_joins=max_num_joins,
        max_num_v1_joins=max_num_v1_joins,
        max_num_predicates=max_num_predicates,
        max_num_predicates_uris=max_num_predicates_uris
    )

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error " + workload_name + ":")
    print_qerror(preds_test_unnorm, label)

    # Write predictions
    file_name = "results/predictions_" + workload_name + ".csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        for i in range(len(preds_test_unnorm)):
            f.write(str(preds_test_unnorm[i]) + "," + str(label[i]) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testset", help="synthetic, scale, or job-light")
    parser.add_argument("--delimiter_col", help="delimiter for column", type=str, default="~")
    parser.add_argument("--queries_data", help="url for dataset of training queries", type=str, default='data/train')
    parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
    parser.add_argument("--samples", help="number of training queries (default: 10000)", type=int, default=10000)
    parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=10)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")
    args = parser.parse_args()
    train_and_predict(args.testset, args.queries_data, args.queries, args.samples, args.epochs, args.batch, args.hid, args.cuda, args.delimiter_col)

if __name__ == "__main__":
    main()

