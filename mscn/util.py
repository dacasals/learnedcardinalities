import numpy as np


# Helper functions for data processing

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_all_column_names(predicates):
    column_names = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                column_names.add(column_name)
    return column_names

def get_all_column_op_uris_names(predicates):
    column_names = set()
    op_names = set()
    uri_names = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 4:

                column_name = predicate[0]
                column_names.add(column_name)

                op_name = predicate[1]
                op_names.add(op_name)

                uri_name = predicate[2]
                if int(predicate[3]) > 3:
                    uri_names.add(uri_name)

    return column_names, op_names, uri_names

def get_sets_joins(joins):
    table_injoin_names = set()
    types_join_names = set()
    for join_row in joins:
        for join in join_row:
            if len(join) == 3:
                table_from_name = join[0]
                table_injoin_names.add(table_from_name)

                table_to_name = join[1]
                table_injoin_names.add(table_to_name)

                types_join_name = join[2]
                types_join_names.add(types_join_name)

    return table_injoin_names, types_join_names

def get_all_table_names(tables):
    table_names = set()
    for query in tables:
        for table in query:
            table_names.add(table)
    return table_names


def get_all_operators(predicates):
    operators = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                operator = predicate[1]
                operators.add(operator)
    return operators


def get_all_joins(joins):
    join_set = set()
    for query in joins:
        for join in query:
            join_set.add(join)
    return join_set


def idx_to_onehot(idx, num_elements):
    onehot = np.zeros(num_elements, dtype=np.float32)
    onehot[idx] = 1.
    return onehot


def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing


def get_min_max_vals(predicates, column_names):
    min_max_vals = {t: [float('inf'), float('-inf')] for t in column_names}
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                val = float(predicate[2])
                if val < min_max_vals[column_name][0]:
                    min_max_vals[column_name][0] = val
                if val > min_max_vals[column_name][1]:
                    min_max_vals[column_name][1] = val
    return min_max_vals


def normalize_data(val, column_name, column_min_max_vals):
    min_val = float(column_min_max_vals[column_name][0])
    max_val = float(column_min_max_vals[column_name][1])
    val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return np.array(val_norm, dtype=np.float32)


def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(l)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype=np.int64)


def encode_samples(tables, samples, table2vec):
    samples_enc = []
    for i, query in enumerate(tables):
        samples_enc.append(list())
        for j, table in enumerate(query):
            sample_vec = []
            # Append table one-hot vector
            sample_vec.append(table2vec[table])
            # Append bit vector
            sample_vec.append(samples[i][j])
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)
    return samples_enc


def encode_tables(tables, table2vec):
    samples_enc = []
    for i, query in enumerate(tables):
        samples_enc.append(list())
        for j, table in enumerate(query):
            sample_vec = []
            # Append table one-hot vector
            sample_vec.append(table2vec[table])
            # Append bit vector
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)
    return samples_enc

def encode_joins_v1(tables, join_v1_2vec, types_join_v1_2vec):
    samples_enc = []
    for i, query in enumerate(tables):
        samples_enc.append(list())
        for j, listData in enumerate(query):
            sample_vec = []
            # Append table one-hot vector
            if len(listData)  == 1:
                #Adding zeros for a join empty as JoinFrom + JoinTo + Types
                len_total = len(join_v1_2vec) * 2 + len(types_join_v1_2vec)
                sample_vec.append(np.zeros(len_total))
            else:
                sample_vec.append(join_v1_2vec[listData[0]])
                sample_vec.append(join_v1_2vec[listData[1]])
                sample_vec.append(types_join_v1_2vec[listData[2]])
            # Append bit vector
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)
    return samples_enc

def encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec):
    predicates_enc = []
    joins_enc = []
    for i, query in enumerate(predicates):
        predicates_enc.append(list())
        joins_enc.append(list())
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                norm_val = normalize_data(val, column, column_min_max_vals)

                pred_vec = []
                pred_vec.append(column2vec[column])
                pred_vec.append(op2vec[operator])
                pred_vec.append(norm_val)
                pred_vec = np.hstack(pred_vec)
            else:
                pred_vec = np.zeros((len(column2vec) + len(op2vec) + 1))

            predicates_enc[i].append(pred_vec)

        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i].append(join_vec)
    return predicates_enc, joins_enc

def get_column_min_max_vals_preds(predicates):
    column_min_max_vals = {}
    for i, query in enumerate(predicates):
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                # norm_val = normalize_data(val, column, column_min_max_vals)
                if column in column_min_max_vals:
                    # If current min is minor of val set as min
                    if column_min_max_vals[column][0] > val:
                        column_min_max_vals[column][0] = val
                    # If current max is major of val set as max
                    if column_min_max_vals[column][1] < val:
                        column_min_max_vals[column][1] = val

                else:
                    #Setting new entry with val as min and max.
                    column_min_max_vals[column] = [val,val]
    return column_min_max_vals


def get_column_min_max_cards_predsuris(predicates):
    """
    Get Column min max cardinalities for predicates
    :param predicates:
    :return:
    """
    column_min_max_cards = {}
    for i, query in enumerate(predicates):
        for predicate in query:
            if len(predicate) == 4:
                # Proper predicate
                column = predicate[0]
                card = predicate[3]
                # norm_val = normalize_data(val, column, column_min_max_vals)
                if column in column_min_max_cards:
                    # If current min is minor of val set as min
                    if column_min_max_cards[column][0] > card:
                        column_min_max_cards[column][0] = card
                    # If current max is major of val set as max
                    if column_min_max_cards[column][1] < card:
                        column_min_max_cards[column][1] = card

                else:
                    #Setting new entry with val as min and max.
                    column_min_max_cards[column] = [card,card]
    return column_min_max_cards

def encode_sparql_data(column_min_max_vals, column_min_max_cards, predicates, predicates_uris, joins, column2vec, op2vec, join2vec, column2uris_vec, op2uris_vec, uris2uris_vec):
    """
    Se codifica la dara utilizando los one_hot_vectors y la data
    :param column_min_max_vals:
    :param column_min_max_cards:
    :param predicates:
    :param predicates_uris:
    :param joins:
    :param column2vec:
    :param op2vec:
    :param join2vec:
    :param column2uris_vec:
    :param op2uris_vec:
    :param uris2uris_vec:
    :return:
    """
    predicates_enc = []
    predicates_uris_enc = []
    joins_enc = []
    # column_min_max_vals = {} If generated in other method
    for i, query in enumerate(predicates):
        predicates_enc.append(list())
        joins_enc.append(list())
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = float(predicate[2])
                norm_val = normalize_data(val, column, column_min_max_vals)
                # if column in column_min_max_vals:
                #     # If current min is minor of val set as min
                #     if column_min_max_vals[column][0] > val:
                #         column_min_max_vals[column][0] = val
                #     # If current max is major of val set as max
                #     if column_min_max_vals[column][1] < val:
                #         column_min_max_vals[column][1] = val
                #
                # else:
                #     #Setting new entry with val as min and max.
                #     column_min_max_vals[column] = [val,val]
                pred_vec = []
                pred_vec.append(column2vec[column])
                pred_vec.append(op2vec[operator])
                pred_vec.append(norm_val)
                pred_vec = np.hstack(pred_vec)
            else:
                # pred_vec = np.zeros((len(column2vec) + len(op2vec)))
                pred_vec = np.zeros((len(column2vec) + len(op2vec) + 1))

            predicates_enc[i].append(pred_vec)

        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i].append(join_vec)

    for j, query2 in enumerate(predicates_uris):
        predicates_uris_enc.append(list())
        #predicates uris <col, op, uri>
        for predicate in query2:
            if len(predicate) == 4:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                cardinality_tpf = float(predicate[3])
                norm_card = normalize_data(cardinality_tpf, column, column_min_max_cards)

                pred_uri_vec = []
                pred_uri_vec.append(column2uris_vec[column])
                pred_uri_vec.append(op2uris_vec[operator])
                if val in uris2uris_vec:
                    pred_uri_vec.append(uris2uris_vec[val])
                else:
                    pred_uri_vec.append(np.zeros((len(uris2uris_vec))))

                pred_uri_vec.append(norm_card)
                pred_uri_vec = np.hstack(pred_uri_vec)
            else:
                pred_uri_vec = np.zeros((len(column2uris_vec) + len(op2uris_vec) + len(uris2uris_vec) + 1))
            predicates_uris_enc[j].append(pred_uri_vec)
    return predicates_enc, joins_enc, predicates_uris_enc
