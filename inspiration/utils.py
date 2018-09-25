from constants import *
import csv
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from dictionary import Dictionary
from collections import defaultdict

def prepare_data(file, dictionary : Dictionary, question_max_length=30):
    with open(file) as csv_file:
        data = csv.reader(csv_file, delimiter=dictionary.pp_data.csv_delimiter)

        X = []
        X_img_features = []
        Y = []
        X_question_id = []

        for row in data:
            question_id = row[0]
            image_id = row[1]

            X_question_id.append(question_id)

            visual_features = dictionary.pp_data.img_features[dictionary.pp_data.image_to_visual_feat_mapping[image_id]]
            X_img_features.append(visual_features)

            question = row[2]
            question = question.lower().strip().strip('?!.').split()
            question_length = len(question)

            complete_answer = row[3]
            x_question = np.zeros(question_max_length)

            try:
                for i in range(question_max_length):
                    if i < question_length:
                        word = question[i]
                        x_question[i] = dictionary.word2idx.get(word, dictionary.word2idx[dictionary.oov])
                    else:
                        break
                y = dictionary.labels2idx.get(complete_answer, dictionary.labels2idx[dictionary.oov])

                Y.append(y)
                X.append(x_question)
            except Exception as e:
                print(str(e))
                pass

        X_return = np.array(X)
        X_img_features = np.array(X_img_features)
        Y_return = np_utils.to_categorical(Y, len(dictionary.labels2idx))


        return (X_return, X_img_features, Y_return, Y, X_question_id)


def analyse_results(inputs, predictions, answers, question_ids, model : Sequential, dictionary : Dictionary, accuracy, model_name, model_type, save_statistics=True):
    results = build_list_of_qpa_dictionaries(inputs, predictions, answers, question_ids, dictionary, model_type)
    #statistics = get_statistics(results)

    if save_statistics:
        np.save(hyper_parameter_folder + final_model_folder + 'FINAL-VISUAL-LSTM-BOW-acc=' + str(accuracy) + ' ' + model_name, results)


def get_statistics(results):
    statistics = dict()
    statistics['total_number_of_results'] = len(results)
    statistics['top1'] = 0
    statistics['top5'] = 0

    statistics['per_type_of_question'] = dict()
    statistics['answer_type'] = dict()

    statistics['number_of_multiple_choice_questions'] = dict()
    statistics['number_of_multiple_choice_questions']['total'] = 0
    statistics['number_of_multiple_choice_questions']['top1'] = 0
    statistics['number_of_multiple_choice_questions']['top5'] = 0


    for result in results:
        statistics['top1'] += 1 if result['prediction'] == result['answer'] else 0
        statistics['top5'] += 1 if result['prediction'] in result['top5'] else 0

        if result['question_type'] not in statistics['per_type_of_question'].keys():
            statistics['per_type_of_question'][result['question_type']] = dict()
            statistics['per_type_of_question'][result['question_type']]['total'] = 1
            statistics['per_type_of_question'][result['question_type']]['top1'] = 1 if result['prediction'] == result['answer'] else 0
            statistics['per_type_of_question'][result['question_type']]['top5'] = 1 if result['prediction'] in result['top5'] else 0
        else:
            statistics['per_type_of_question'][result['question_type']]['total'] += 1
            statistics['per_type_of_question'][result['question_type']]['top1'] += 1 if result['prediction'] == result[
                'answer'] else 0
            statistics['per_type_of_question'][result['question_type']]['top5'] += 1 if result['prediction'] in result[
                'top5'] else 0

        if result['answer_type'] not in statistics['answer_type'].keys():
            statistics['answer_type'][result['answer_type']] = dict()
            statistics['answer_type'][result['answer_type']]['total'] = 0
            statistics['answer_type'][result['answer_type']]['top1'] = 0
            statistics['answer_type'][result['answer_type']]['top5'] = 0

        statistics['answer_type'][result['answer_type']]['total'] += 1
        statistics['answer_type'][result['answer_type']]['top1'] += 1 if result['prediction'] == result['answer'] else 0
        statistics['answer_type'][result['answer_type']]['top5'] += 1 if result['prediction'] in result['top5'] else 0

        statistics['number_of_multiple_choice_questions']['total'] += 1
        statistics['number_of_multiple_choice_questions']['top1'] += 1 if result['prediction'] == result['answer'] else 0
        statistics['number_of_multiple_choice_questions']['top5'] += 1 if result['prediction'] in result['top5'] else 0

    return statistics


# returns list of dictionaries. Dictionary format is ['img_id', 'question', 'prediction', 'answer', 'correct']
def build_list_of_qpa_dictionaries(inputs, predictions, answers, question_ids, dictionary : Dictionary, model_type):
    N = len(predictions)
    results = list()

    test_data = dictionary.pp_data.test_data
    val_data = dictionary.pp_data.val_data

    skipped_questions = 0

    for i in range(N):
        predictions_for_question = predictions[i]
        prediction_idx_for_question = np.argmax(predictions_for_question)

        question_id = str(question_ids[i])
        answer = int(answers[i])

        question_embed = inputs[i]
        question = ''

        for j in range(10):
            idx = question_embed[j]
            word = dictionary.idx2word[int(idx)]

            question += word

            if j < 9 and not question_embed[j + 1] == 0:
                question += ' '
            else:
                break

        top5predictions = predictions_for_question.argsort()[-5:][::-1]

        found_question = False

        if question_id in test_data.keys():
            found_question = True
            question_info = test_data[question_id]['annotations']

        elif question_id in val_data.keys():
            found_question = True
            question_info = val_data[question_id]['annotations']

        if not found_question:
            skipped_questions += 1
            continue


        result = dict()

        result['image_id'] = question_info['image_id']
        result['question_id'] = question_id
        result['question_name'] = question
        result['question_type'] = question_info['question_type']
        result['question_multiple_choice'] = question_info['multiple_choice_answer']
        result['answer_type'] = question_info['answer_type']
        result['prediction'] = dictionary.idx2labels[int(prediction_idx_for_question)]
        result['answer'] = dictionary.idx2labels[int(answer)]
        result['top5'] = [dictionary.idx2labels[int(prediction)] for prediction in top5predictions]

        results.append(result)

    np.save(data_folder + results_write_file + model_type, results)

    if (skipped_questions != 0):
        print('SKIPPED QUESTIONS, INVESTIGATE: ' + str(skipped_questions))

    return results
