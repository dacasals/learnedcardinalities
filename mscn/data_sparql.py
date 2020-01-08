import csv
import torch
from torch.utils.data import dataset

from mscn.util import *


def load_data(file_name, num_materialized_samples, with_header=True, delimiter='~'):
    joins = []
    joins_v1 = []
    predicates = []
    predicates_uris = []
    tables = []
    samples = []
    label = []
    header_ignored=False
    tables_pos = 0
    cardinality_pos = 3
    joins_pos = 1
    predicates_pos = 2
    predicatesuri_pos = 3

    # Load queries
    with open(file_name, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=delimiter))
        for row in data_raw:
            if with_header and not header_ignored:
                tables_pos = row.index("tables")
                joins_pos = row.index("joins")
                joins_v1_pos = row.index("joins_v1")
                predicates_pos = row.index("predicates_v2int")
                predicatesuri_pos = row.index("pred_v2uri_cardinality")
                cardinality_pos = row.index("cardinality")
                header_ignored = True
                continue
            tables.append(list(filter(None, row[tables_pos].split(','))))

            joins.append(list(filter(None, row[joins_pos].split(','))))

            joins_v1.append(list(filter(None, row[joins_v1_pos].split(','))))

            predicates.append(list(filter(None, row[predicates_pos].split(','))))

            predicates_uris.append(list(filter(None, row[predicatesuri_pos].split(','))))
            if int(row[cardinality_pos]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            label.append(row[cardinality_pos])
    print("Loaded queries")

    # Load bitmaps
    if num_materialized_samples > 0:
        num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
        with open(file_name[:-4] + ".bitmaps", 'rb') as f:
            for i in range(len(tables)):
                four_bytes = f.read(4)
                if not four_bytes:
                    print("Error while reading 'four_bytes'")
                    exit(1)
                num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
                bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
                for j in range(num_bitmaps_curr_query):
                    # Read bitmap
                    bitmap_bytes = f.read(num_bytes_per_bitmap)
                    if not bitmap_bytes:
                        print("Error while reading 'bitmap_bytes'")
                        exit(1)
                    bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
                samples.append(bitmaps)
        print("Loaded bitmaps")

    # Split predicates
    joins_v1 = [list(chunks(d, 3)) for d in joins_v1]
    predicates = [list(chunks(d, 3)) for d in predicates]
    predicates_uris = [list(chunks(d, 4)) for d in predicates_uris]

    return joins, joins_v1, predicates, predicates_uris, tables, samples, label


def load_and_encode_train_data(url_queries, num_queries, num_materialized_samples, delimiter):
    file_name_queries = url_queries
    # file_name_column_min_max_vals = "data/column_min_max_vals.csv"

    joins, joins_v1, predicates, predicates_uris, tables, samples, label = load_data(file_name_queries, num_materialized_samples,delimiter=delimiter)

    # Get column name dict for col,operator,val
    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)

    # Get column name dict for col,operator,uri
    column_uris_names, op_uris_names, uris_set_names= get_all_column_op_uris_names(predicates_uris)

    column2uris_vec, idx2column_uris = get_set_encoding(column_uris_names)
    op2uris_vec, idx2op_uris = get_set_encoding(op_uris_names)
    uris2uris_vec, idx2uris_uris = get_set_encoding(uris_set_names)

    # Get table name dict
    table_names = get_all_table_names(tables)
    table2vec, idx2table = get_set_encoding(table_names)

    # Get operator name dict
    operators = get_all_operators(predicates)
    op2vec, idx2op = get_set_encoding(operators)

    # Get join name dict
    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    # Get tables name dict for joins_v1
    table_injoin_names, types_injoin_names = get_sets_joins(joins_v1)
    join_v1_2vec, idx2join_v1_ = get_set_encoding(table_injoin_names)
    types_injoin_2vec, idx2types_injoin = get_set_encoding(types_injoin_names)

    # Get feature encoding and proper normalization
    samples_enc  = encode_tables(tables, table2vec)
    joins_v1_enc = encode_joins_v1(joins_v1, join_v1_2vec, types_injoin_2vec)

    #Getting the encoded data form predicates, predicates_uris joins and additionaly column_min_max_vals of predicates.
    predicates_enc, joins_enc, predicates_uris_enc,column_min_max_vals = encode_sparql_data(predicates, predicates_uris, joins, column2vec, op2vec, join2vec,column2uris_vec, op2uris_vec, uris2uris_vec)
    label_norm, min_val, max_val = normalize_labels(label)

    # Split in training and validation samples
    num_train = int(num_queries * 0.9)
    num_test = num_queries - num_train

    samples_train = samples_enc[:num_train]
    predicates_train = predicates_enc[:num_train]
    predicates_uri_train = predicates_uris_enc[:num_train]
    joins_train = joins_enc[:num_train]
    joins_v1_train = joins_v1_enc[:num_train]
    labels_train = label_norm[:num_train]

    samples_test = samples_enc[num_train:num_train + num_test]
    predicates_test = predicates_enc[num_train:num_train + num_test]
    predicates_uri_test = predicates_uris_enc[num_train:num_train + num_test]
    joins_test = joins_enc[num_train:num_train + num_test]
    joins_v1_test = joins_enc[num_train:num_train + num_test]
    labels_test = label_norm[num_train:num_train + num_test]

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_test)))

    max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
    max_num_v1_joins = max(max([len(j) for j in joins_v1_train]), max([len(j) for j in joins_v1_test]))

    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))
    max_num_predicates_uris = max(max([len(p) for p in predicates_uri_train]), max([len(p) for p in predicates_uri_test]))

    dicts = [table2vec, column2vec, op2vec, join2vec, join_v1_2vec, column2uris_vec, op2uris_vec]
    train_data = [samples_train, predicates_train, joins_train, joins_v1_train, predicates_uri_train]
    test_data = [samples_test, predicates_test, joins_test,joins_v1_test, predicates_uri_test]
    return dicts,\
           column_min_max_vals, \
           min_val, \
           max_val, \
           labels_train, \
           labels_test, \
           max_num_joins, \
           max_num_v1_joins, \
           max_num_predicates, \
           max_num_predicates_uris, \
           train_data, \
           test_data


def make_dataset(samples, predicates, joins, joins_v1, predicates_uri, labels, max_num_joins, max_num_v1_joins, max_num_predicates, max_num_predicates_uris):
    """
    Add zero-padding and wrap as tensor dataset.
    :param samples:
    :param predicates:
    :param joins:
    :param joins_v1: New version of joins in the way zeros for predFrom + zeros from predTo + zeros from type join
    :param predicates_uri:
    :param labels:
    :param max_num_joins:
    :param max_num_v1_joins:
    :param max_num_predicates:
    :return:
    """

    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor = np.vstack(sample)
        # num_pad = max_num_joins + 1 - sample_tensor.shape[0]
        num_pad = max_num_v1_joins + 1 - sample_tensor.shape[0]

        sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
        sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
        sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    sample_masks = torch.FloatTensor(sample_masks)

    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    predicate_masks = torch.FloatTensor(predicate_masks)

    predicate_uri_masks = []
    predicate_uri_tensors = []

    for predicate in predicates_uri:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        predicate_uri_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_uri_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_uri_tensors = np.vstack(predicate_uri_tensors)
    predicate_uri_tensors = torch.FloatTensor(predicate_uri_tensors)
    predicate_uri_masks = np.vstack(predicate_uri_masks)
    predicate_uri_masks = torch.FloatTensor(predicate_uri_masks)

    join_masks = []
    join_tensors = []
    # for join in joins:
    for join in joins_v1:
        join_tensor = np.vstack(join)
        # num_pad = max_num_joins - join_tensor.shape[0]
        num_pad = max_num_v1_joins - join_tensor.shape[0]
        join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
        join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
        join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
        join_tensors.append(np.expand_dims(join_tensor, 0))
        join_masks.append(np.expand_dims(join_mask, 0))
    join_tensors = np.vstack(join_tensors)
    join_tensors = torch.FloatTensor(join_tensors)
    join_masks = np.vstack(join_masks)
    join_masks = torch.FloatTensor(join_masks)

    target_tensor = torch.FloatTensor(labels)

    return dataset.TensorDataset(
        sample_tensors,
        predicate_tensors,
        join_tensors,
        target_tensor,
        sample_masks,
        predicate_masks,
        join_masks,
        predicate_uri_tensors,
        predicate_uri_masks
    )

def get_train_datasets(url_queries, num_queries, num_materialized_samples, delimiter):
    dicts, \
    column_min_max_vals, \
    min_val, max_val, \
    labels_train, \
    labels_test, \
    max_num_joins, \
    max_num_v1_joins, \
    max_num_predicates, \
    max_num_predicates_uris, \
    train_data, \
    test_data \
        = load_and_encode_train_data \
            (
            url_queries,
            num_queries,
            num_materialized_samples,
            delimiter=delimiter
        )

    train_dataset = make_dataset(
        *train_data,
        labels=labels_train,
        max_num_joins=max_num_joins,
        max_num_v1_joins=max_num_v1_joins,
        max_num_predicates=max_num_predicates,
        max_num_predicates_uris=max_num_predicates_uris)
    print("Created TensorDataset for training data")

    test_dataset = make_dataset(
        *test_data,
        labels=labels_test,
        max_num_joins=max_num_joins,
        max_num_predicates=max_num_predicates
    )
    print("Created TensorDataset for validation data")
    return dicts, \
           min_val, \
           max_val, \
           labels_train, \
           labels_test, \
           max_num_joins, \
           max_num_predicates, \
           train_dataset, \
           test_dataset
