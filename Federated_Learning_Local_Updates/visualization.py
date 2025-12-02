import matplotlib.pyplot as plt

def plot_training_progress(all_train_loss, all_accuracy, NUM_EPOCHS, LOCAL_ITERS, BATCH_SIZE, NUM_CLIENTS, DATASET, accuracy, NUM_MALICIOUS_CLIENT, EPSILON, DP_TYPE, DELTA, ATTACT_TYPE, LABEL_FLIP_PER, AGGREGATE_METHOD,save_path=None):
    plt.figure(figsize=(16, 4))

    if ATTACT_TYPE == "ModelP" or ATTACT_TYPE == "No_Attack":
        if DP_TYPE == "No_DP":
            plt.suptitle(
                'Federated Learning trained with {} | Total Communication Rounds: {}\n'
                'Final Test Accuracy: {}\n'
                'Number of clients: {}, Number of malicious Client: {}, Attack Type: {} \n'
                'Local iterations: {}, Batch size: {}, Aggregation Method: {}  \n'
                'No Differencial Privacy Applied'.format(
                    DATASET, len(all_accuracy), accuracy*100, NUM_CLIENTS, NUM_MALICIOUS_CLIENT, ATTACT_TYPE, LOCAL_ITERS, BATCH_SIZE, AGGREGATE_METHOD,
                )
            )
        else:
            plt.suptitle(
                'Federated Learning trained with {} | Total Communication Rounds: {}\n'
                'Final Test Accuracy: {}\n'
                'Number of clients: {}, Number of malicious Client: {}, Attack Type: {} \n'
                'Local iterations: {}, Batch size: {}, Aggregation Method: {}  \n'
                'DP_TYPE: {}, Epsilon: {}, Delta: {}'.format(
                    DATASET, len(all_accuracy), accuracy*100, NUM_CLIENTS, NUM_MALICIOUS_CLIENT, ATTACT_TYPE, LOCAL_ITERS, BATCH_SIZE, AGGREGATE_METHOD, DP_TYPE, EPSILON, DELTA,
                )
            )
    else:
        if DP_TYPE == "No_DP":
            plt.suptitle(
                'Federated Learning trained with {} | Total Communication Rounds: {}\n'
                'Final Test Accuracy: {}\n'
                'Number of clients: {}, Number of malicious Client: {}, Attack Type: {}, Label Flip Percentage: {} \n'
                'Local iterations: {}, Batch size: {}, Aggregation Method: {}  \n'
                'No Differencial Privacy Applied'.format(
                    DATASET, len(all_accuracy), accuracy*100, NUM_CLIENTS, NUM_MALICIOUS_CLIENT, ATTACT_TYPE, LABEL_FLIP_PER, LOCAL_ITERS, BATCH_SIZE, AGGREGATE_METHOD,
                )
            )
        else:
            plt.suptitle(
                'Federated Learning trained with {} | Total Communication Rounds: {}\n'
                'Final Test Accuracy: {}\n'
                'Number of clients: {}, Number of malicious Client: {}, Attack Type: {}, Label Flip Percentage: {} \n'
                'Local iterations: {}, Batch size: {}, Aggregation Method: {}  \n'
                'DP_TYPE: {}, Epsilon: {}, Delta: {}'.format(
                    DATASET, len(all_accuracy), accuracy*100, NUM_CLIENTS, NUM_MALICIOUS_CLIENT, ATTACT_TYPE, LABEL_FLIP_PER, LOCAL_ITERS, BATCH_SIZE, AGGREGATE_METHOD, DP_TYPE, EPSILON, DELTA,
                )
            )

    plt.subplot(121)
    plt.plot(list(range(len(all_train_loss))), all_train_loss)
    plt.title('Global Loss over Communication Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Global Loss')

    plt.subplot(122)
    plt.plot(list(range(len(all_accuracy))), all_accuracy)
    plt.title('Global Accuracy over Communication Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Global Accuracy')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = save_path.replace(".", "_")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    #plt.show()
