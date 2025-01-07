import csv
import json
import logging
import os
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from nebula.core.engine import Engine


def save_data(
    scenario,
    type_data,
    source_ip,
    addr,
    num_round=None,
    time=None,
    type_message=None,
    current_round=None,
    fraction_changed=None,
    total_params=None,
    changed_params=None,
    threshold=None,
    changes_record=None,
    message_id_decoded=None,
    latency=None,
):
    """
    Save data between nodes and aggregated models.

    Args:
        source_ip (str): Source IP address.
        addr (str): Destination IP address.
        round (int): Round number.
        time (float): Time taken to process the data.
    """

    source_ip = source_ip.split(":")[0]
    addr = addr.split(":")[0]

    try:
        combined_data = {}

        if type_data == "time_message":
            combined_data["time_message"] = {
                "time": time,
                "type_message": type_message,
                "round": num_round,
                "current_round": current_round,
            }
        elif type_data == "fraction_of_params_changed":
            combined_data["fraction_of_params_changed"] = {
                "total_params": total_params,
                "changed_params": changed_params,
                "fraction_changed": fraction_changed,
                "threshold": threshold,
                "changes_record": changes_record,
                "round": num_round,
            }
        elif type_data == "model_arrival_latency":
            combined_data["model_arrival_latency"] = {
                "latency": latency,
                "round": num_round,
            }

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = f"{addr}_storing_{source_ip}_info.json"
        full_file_path = os.path.join(script_dir, scenario, file_name)
        os.makedirs(os.path.dirname(full_file_path), exist_ok=True)

        all_metrics = []
        if os.path.exists(full_file_path):
            with open(full_file_path) as existing_file:
                try:
                    all_metrics = json.load(existing_file)
                except json.JSONDecodeError:
                    logging.exception(f"JSON decode error in file: {full_file_path}")
                    all_metrics = []

        all_metrics.append(combined_data)

        with open(full_file_path, "w") as json_file:
            json.dump(all_metrics, json_file, indent=4)

    except Exception:
        logging.exception("Error saving data")


class Reputation:
    """
    Class to define the reputation of a participant.
    """

    reputation_history: ClassVar[dict] = {}
    time_message_history: ClassVar[dict] = {}
    neighbor_reputation_history: ClassVar[dict] = {}
    fraction_changed_history: ClassVar[dict] = {}
    messages_time_message: ClassVar[list] = []
    previous_threshold_time_message: ClassVar[dict] = {}
    previous_std_dev_time_message: ClassVar[dict] = {}
    messages_model_arrival_latency: ClassVar[dict] = {}
    model_arrival_latency_history: ClassVar[dict] = {}
    previous_percentile_25_time_message: ClassVar[dict] = {}
    previous_percentile_85_time_message: ClassVar[dict] = {}

    def __init__(self, engine: "Engine"):
        self._engine = engine
        self.model_arrival_latency_data = {}

    @property
    def engine(self):
        return self._engine

    def calculate_reputation(self, scenario, log_dir, id_node, addr, nei, current_round=None):
        """
        Calculate the reputation of each participant based on the data stored.

        Args:
            scenario (str): Scenario name.
        """
        logging.info(f"id_node: {id_node}, addr: {addr}, nei: {nei}")
        addr = addr.split(":")[0].strip()
        nei = nei.split(":")[0].strip()

        messages_time_message_normalized = 0
        messages_time_message_count = 0
        avg_messages_time_message_normalized = 0
        fraction_score = 0
        fraction_score_normalized = 0
        fraction_score_asign = 0
        messages_model_arrival_latency_normalized = 0
        avg_model_arrival_latency = 0
        fraction_neighbors_scores = None

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_name = f"{addr}_storing_{nei}_info.json"
            full_file_path = os.path.join(script_dir, scenario, file_name)
            os.makedirs(os.path.dirname(full_file_path), exist_ok=True)

            if os.path.exists(full_file_path) and os.path.getsize(full_file_path) > 0:
                with open(full_file_path) as json_file:
                    all_metrics = json.load(json_file)
                    for metric in all_metrics:
                        if "time_message" in metric:
                            round_time = metric["time_message"]["round"]
                            current_round_time = metric["time_message"]["current_round"]
                            time = metric["time_message"]["time"]
                            type_message = metric["time_message"]["type_message"]
                            previous_round_time = current_round - 1
                            # logging.info(f"Current round time: {current_round_time}, previous round time: {previous_round_time}")
                            if round_time == previous_round_time:
                                # logging.info(f"Time message: {time}, type message: {type_message}, round: {round_time}")
                                Reputation.messages_time_message.append({
                                    "time_message": time,
                                    "type_message": type_message,
                                    "round": round_time,
                                    "current_round": current_round_time,
                                    "key": (addr, nei),
                                })
                        if "fraction_of_params_changed" in metric:
                            round_fraction = metric["fraction_of_params_changed"]["round"]
                            total_params = metric["fraction_of_params_changed"]["total_params"]
                            changed_params = metric["fraction_of_params_changed"]["changed_params"]
                            fraction_changed = metric["fraction_of_params_changed"]["fraction_changed"]
                            threshold = metric["fraction_of_params_changed"]["threshold"]
                            changes_record = metric["fraction_of_params_changed"]["changes_record"]
                            if round_fraction == current_round:
                                fraction_score_normalized = Reputation.analyze_anomalies(
                                    addr,
                                    nei,
                                    round_fraction,
                                    current_round,
                                    fraction_changed,
                                    threshold,
                                    changes_record,
                                    changed_params,
                                    total_params,
                                    scenario,
                                )
                                # logging.info(f"Fraction score normalized: {fraction_score_normalized}")
                        if "model_arrival_latency" in metric:
                            round_latency = metric["model_arrival_latency"]["round"]
                            latency = metric["model_arrival_latency"]["latency"]
                            if round_latency == current_round:
                                messages_model_arrival_latency_normalized = Reputation.manage_model_arrival_latency(
                                    round_latency, addr, nei, latency, scenario, self.model_arrival_latency_data
                                )
                                logging.info(
                                    f"messages_model_arrival_latency_normalized: {messages_model_arrival_latency_normalized}"
                                )

                    similarity_file = os.path.join(log_dir, f"participant_{id_node}_similarity.csv")
                    similarity_reputation = Reputation.read_similarity_file(similarity_file, nei)

                    if messages_model_arrival_latency_normalized >= 0:
                        avg_model_arrival_latency = Reputation.save_model_arrival_latency_history(
                            addr, nei, messages_model_arrival_latency_normalized, current_round
                        )
                        if avg_model_arrival_latency is None and current_round > 4:
                            avg_model_arrival_latency = Reputation.model_arrival_latency_history[(addr, nei)][
                                current_round - 1
                            ]["score"]
                            # logging.info(f"Avg model_arrival_latency latency is None and current_round = {current_round}, model_arrival_latency: {avg_model_arrival_latency}")

                    if Reputation.messages_time_message is not None:
                        messages_time_message_normalized, messages_time_message_count = (
                            Reputation.manage_metric_time_message(
                                Reputation.messages_time_message, addr, nei, current_round, scenario
                            )
                        )
                        # logging.info(f"Messages time_message normalized: {messages_time_message_normalized}")
                        avg_messages_time_message_normalized = Reputation.save_time_message_history(
                            addr, nei, messages_time_message_normalized, current_round
                        )
                        # logging.info(f"Avg messages time_message normalized: {avg_messages_time_message_normalized}")
                        if avg_messages_time_message_normalized is None and current_round >= 4:
                            avg_messages_time_message_normalized = Reputation.time_message_history[(addr, nei)][
                                current_round - 1
                            ]["avg_time_message"]
                            # (f"Avg messages time_message is None and curret_round = {current_round}, avg_messages_time_message_normalized: {avg_messages_time_message_normalized}")

                    if current_round >= 5:
                        if fraction_score_normalized > 0:
                            key_previous_round = (addr, nei, current_round - 1) if current_round - 1 > 0 else None
                            fraction_previous_round = None

                            if (
                                key_previous_round is not None
                                and key_previous_round in Reputation.fraction_changed_history
                            ):
                                fraction_score = Reputation.fraction_changed_history[key_previous_round].get(
                                    "fraction_score"
                                )
                                fraction_previous_round = fraction_score if fraction_score is not None else None
                                # logging.info(f"Fraction score previous round: {fraction_previous_round}")

                            if fraction_previous_round is not None:
                                fraction_score_asign = (fraction_score_normalized + fraction_previous_round) / 2
                                # logging.info(f"Fraction score normalized: {fraction_score_asign}")
                                Reputation.fraction_changed_history[(addr, nei, current_round)]["fraction_score"] = (
                                    fraction_score_asign
                                )
                            else:
                                fraction_score_asign = fraction_score_normalized
                                # logging.info(f"Fraction score normalized: {fraction_score_asign}")
                                Reputation.fraction_changed_history[(addr, nei, current_round)]["fraction_score"] = (
                                    fraction_score_asign
                                )
                        else:
                            # logging.info(f"No fraction score to assign")
                            fraction_previous_round = None
                            key_previous_round = (addr, nei, current_round - 1) if current_round - 1 > 0 else None
                            if (
                                key_previous_round is not None
                                and key_previous_round in Reputation.fraction_changed_history
                            ):
                                fraction_score = Reputation.fraction_changed_history[key_previous_round].get(
                                    "fraction_score"
                                )
                                fraction_previous_round = fraction_score if fraction_score is not None else None
                                # logging.info(f"Fraction score previous round: {fraction_previous_round}")

                            if fraction_previous_round is not None:
                                fraction_score_asign = fraction_previous_round - (fraction_previous_round * 0.1)
                                # logging.info(f"Fraction score normalized: {fraction_score_asign}")

                            if fraction_neighbors_scores is None:
                                fraction_neighbors_scores = {}
                            # logging.info(f"fraction_neighbors_scores: {fraction_neighbors_scores}")

                            if fraction_previous_round is None:
                                for key, value in Reputation.fraction_changed_history.items():
                                    score = value.get("fraction_score")
                                    # logging.info(f"Score: {score}")
                                    if score is not None:
                                        fraction_neighbors_scores[key] = score

                            if fraction_neighbors_scores:
                                fraction_score_asign = np.mean(list(fraction_neighbors_scores.values()))
                            else:
                                fraction_score_asign = 0  # O un valor predeterminado adecuado
                    else:
                        fraction_score_asign = 0

                    # Weights for each metric
                    if current_round is not None:
                        if current_round >= 5:  # only for rounds 5 and later
                            # Weight to scenarios with delay
                            # weight_to_similarity = 0.1
                            # weight_to_fraction = 0.1
                            # weight_to_message_time_message = 0.1
                            # weight_to_model_arrival_latency = 0.7
                            # Weight to scenarios with noise injection
                            # weight_to_similarity = 0.4
                            # weight_to_fraction = 0.4
                            # weight_to_message_time_message = 0.1
                            # weight_to_model_arrival_latency = 0.1
                            # Weight to scenarios with flood
                            weight_to_similarity = 0.1
                            weight_to_fraction = 0.1
                            weight_to_message_time_message = 0.6
                            weight_to_model_arrival_latency = 0.2
                        elif current_round <= 4:  # only for rounds 0, 1, 2, 3
                            weight_to_similarity = 1.0
                            weight_to_fraction = 0.0
                            weight_to_message_time_message = 0.0
                            weight_to_model_arrival_latency = 0.0

                    logging.info(f"Weight to similarity: {weight_to_similarity}")
                    logging.info(f"Weight to fraction: {weight_to_fraction}")
                    logging.info(f"Weight to message_time_message: {weight_to_message_time_message}")
                    logging.info(f"Weight to model_arrival_latency: {weight_to_model_arrival_latency}")

                    # Reputation calculation
                    reputation = (
                        weight_to_message_time_message * avg_messages_time_message_normalized
                        + weight_to_similarity * similarity_reputation
                        + weight_to_fraction * fraction_score_asign
                        + weight_to_model_arrival_latency * avg_model_arrival_latency
                    )

                    # Create graphics to metrics
                    self.create_graphics_to_metrics(
                        messages_time_message_count,
                        avg_messages_time_message_normalized,
                        similarity_reputation,
                        fraction_score_asign,
                        avg_model_arrival_latency,
                        addr,
                        nei,
                        current_round,
                        self.engine.total_rounds,
                        scenario,
                        reputation,
                    )

                    # Save history reputation
                    average_reputation = Reputation.save_reputation_history_in_memory(
                        addr, nei, reputation, current_round
                    )

                    logging.info(f"Average reputation to node {nei}: {average_reputation}")
            return average_reputation
        except Exception as e:
            logging.exception(f"Error calculating reputation. Type: {type(e).__name__}")

    def create_graphics_to_metrics(
        self,
        time_message_count,
        time_message_norm,
        similarity,
        fraction,
        model_arrival_latency,
        addr,
        nei,
        current_round,
        total_rounds,
        scenario,
        reputation,
    ):
        """
        Create graphics to metrics.
        """

        if current_round is not None and current_round < total_rounds:
            model_arrival_latency_dict = {f"R-Model_arrival_latency_reputation/{addr}": {nei: model_arrival_latency}}

            messages_time_message_count_dict = {
                f"R-Count_messages_time_message_reputation/{addr}": {nei: time_message_count}
            }

            messages_time_message_norm_dict = {f"R-time_message_reputation/{addr}": {nei: time_message_norm}}

            similarity_dict = {f"R-Similarity_reputation/{addr}": {nei: similarity}}

            fraction_dict = {f"R-Fraction_reputation/{addr}": {nei: fraction}}

            if messages_time_message_count_dict is not None:
                self.engine.trainer._logger.log_data(messages_time_message_count_dict, step=current_round)

            if messages_time_message_norm_dict is not None:
                self.engine.trainer._logger.log_data(messages_time_message_norm_dict, step=current_round)

            if similarity_dict is not None:
                self.engine.trainer._logger.log_data(similarity_dict, step=current_round)

            if fraction_dict is not None:
                self.engine.trainer._logger.log_data(fraction_dict, step=current_round)

            if model_arrival_latency_dict is not None:
                self.engine.trainer._logger.log_data(model_arrival_latency_dict, step=current_round)

            data = {
                "addr": addr,
                "nei": nei,
                "round": current_round,
                "time_message_count": time_message_count,
                "time_message_norm": time_message_norm,
                "similarity": similarity,
                "fraction": fraction,
                "model_arrival_latency": model_arrival_latency,
                "reputation_without_feedback": reputation,
            }
            Reputation.metrics(scenario, data, addr, nei, "reputation")

    @staticmethod
    def analyze_anomalies(
        addr,
        nei,
        round_num,
        current_round,
        fraction_changed,
        threshold,
        changes_record,
        changed_params,
        total_params,
        scenario,
    ):
        """
        Analyze anomalies in the fraction of parameters changed.
        """
        try:
            key = (addr, nei, round_num)

            if key not in Reputation.fraction_changed_history:
                prev_key = (addr, nei, round_num - 1)
                if round_num > 0 and prev_key in Reputation.fraction_changed_history:
                    previous_data = Reputation.fraction_changed_history[prev_key]
                    fraction_changed = (
                        fraction_changed if fraction_changed is not None else previous_data["fraction_changed"]
                    )
                    threshold = threshold if threshold is not None else previous_data["threshold"]
                else:
                    fraction_changed = fraction_changed if fraction_changed is not None else 0
                    threshold = threshold if threshold is not None else 0

                Reputation.fraction_changed_history[key] = {
                    "fraction_changed": fraction_changed,
                    "threshold": threshold,
                    "fraction_score": None,
                    "fraction_anomaly": False,
                    "threshold_anomaly": False,
                    "mean_fraction": None,
                    "std_dev_fraction": None,
                    "mean_threshold": None,
                    "std_dev_threshold": None,
                }

            # Calcular y almacenar estadísticas solo hasta la ronda 4
            if round_num < 5:
                past_fractions = []
                past_thresholds = []

                for r in range(round_num):
                    past_key = (addr, nei, r)
                    if past_key in Reputation.fraction_changed_history:
                        past_fractions.append(Reputation.fraction_changed_history[past_key]["fraction_changed"])
                        past_thresholds.append(Reputation.fraction_changed_history[past_key]["threshold"])

                if past_fractions:
                    mean_fraction = np.mean(past_fractions)
                    # logging.info(f"Round: {round_num}, Mean fraction: {mean_fraction}")
                    std_dev_fraction = np.std(past_fractions)
                    # logging.info(f"Round: {round_num}, Std dev fraction: {std_dev_fraction}")
                    Reputation.fraction_changed_history[key]["mean_fraction"] = mean_fraction
                    Reputation.fraction_changed_history[key]["std_dev_fraction"] = std_dev_fraction

                if past_thresholds:
                    mean_threshold = np.mean(past_thresholds)
                    # logging.info(f"Round: {round_num}, Mean threshold: {mean_threshold}")
                    std_dev_threshold = np.std(past_thresholds)
                    # logging.info(f"Round: {round_num}, Std dev threshold: {std_dev_threshold}")
                    Reputation.fraction_changed_history[key]["mean_threshold"] = mean_threshold
                    Reputation.fraction_changed_history[key]["std_dev_threshold"] = std_dev_threshold

                return 0
            else:
                fraction_value = 0
                threshold_value = 0
                prev_key = (addr, nei, round_num - 1)
                if prev_key not in Reputation.fraction_changed_history:
                    for i in range(0, round_num + 1):
                        potential_prev_key = (addr, nei, round_num - i)
                        if potential_prev_key in Reputation.fraction_changed_history:
                            prev_key = potential_prev_key
                            break

                if prev_key:
                    mean_fraction_prev = Reputation.fraction_changed_history[prev_key]["mean_fraction"]
                    std_dev_fraction_prev = Reputation.fraction_changed_history[prev_key]["std_dev_fraction"]
                    mean_threshold_prev = Reputation.fraction_changed_history[prev_key]["mean_threshold"]
                    std_dev_threshold_prev = Reputation.fraction_changed_history[prev_key]["std_dev_threshold"]
                    # logging.info(f"Round: {round_num}, Mean fraction: {mean_fraction_prev}, Std dev fraction: {std_dev_fraction_prev}, Mean threshold: {mean_threshold_prev}, Std dev threshold: {std_dev_threshold_prev}")

                    current_fraction = Reputation.fraction_changed_history[key]["fraction_changed"]
                    current_threshold = Reputation.fraction_changed_history[key]["threshold"]
                    # logging.info(f"Round: {round_num}, Current fraction: {current_fraction}, Current threshold: {current_threshold}")

                    # low_mean_fraction_prev = mean_fraction_prev - std_dev_fraction_prev
                    upper_mean_fraction_prev = (mean_fraction_prev) + (std_dev_fraction_prev)
                    # low_mean_threshold_prev = mean_threshold_prev - std_dev_threshold_prev
                    upper_mean_threshold_prev = (mean_threshold_prev) + (std_dev_threshold_prev)
                    # logging.info(f"Round: {round_num}, Upper mean fraction: {upper_mean_fraction_prev}, Upper mean threshold: {upper_mean_threshold_prev}")

                    # fraction_anomaly = not (low_mean_fraction_prev <= current_fraction <= upper_mean_fraction_prev)
                    # threshold_anomaly = not (low_mean_threshold_prev <= current_threshold <= upper_mean_threshold_prev)
                    fraction_anomaly = current_fraction > upper_mean_fraction_prev
                    threshold_anomaly = current_threshold > upper_mean_threshold_prev
                    # logging.info(f"Round: {round_num}, Fraction anomaly: {fraction_anomaly}, Threshold anomaly: {threshold_anomaly}")

                    Reputation.fraction_changed_history[key]["fraction_anomaly"] = fraction_anomaly
                    Reputation.fraction_changed_history[key]["threshold_anomaly"] = threshold_anomaly

                    # Calculate the fraction score
                    k_fraction = std_dev_fraction_prev if std_dev_fraction_prev != 0 else 1
                    # logging.info(f"Round: {round_num}, K fraction: {k_fraction}")
                    k_threshold = std_dev_threshold_prev if std_dev_threshold_prev != 0 else 1
                    # logging.info(f"Round: {round_num}, K threshold: {k_threshold}")
                    fraction_value = (
                        1 - (1 / (1 + np.exp(-k_fraction * (current_fraction - mean_fraction_prev))))
                        if current_fraction is not None and mean_fraction_prev is not None
                        else 0
                    )
                    # logging.info(f"Round: {round_num}, Fraction_value: {fraction_value}")

                    threshold_value = (
                        1 - (1 / (1 + np.exp(-k_threshold * (current_threshold - mean_threshold_prev))))
                        if current_threshold is not None and mean_threshold_prev is not None
                        else 0
                    )
                    # logging.info(f"Round: {round_num}, Threshold_value: {threshold_value}")

                    # if threshold_anomaly:
                    #     fraction_weight = 0.8
                    #     threshold_weight = 0.2
                    # elif fraction_anomaly:
                    #     fraction_weight = 0.4
                    #     threshold_weight = 0.6
                    # else:
                    # fraction_weight = 0.5
                    # threshold_weight = 0.5

                    fraction_weight = 0.5
                    threshold_weight = 0.5

                    fraction_score = fraction_weight * fraction_value + threshold_weight * threshold_value
                    # logging.info(f"Round: {round_num}, Fraction score: {fraction_score}")
                    # Reputation.fraction_changed_history[key]["fraction_score"] = fraction_score

                    # Upload the values to the history
                    Reputation.fraction_changed_history[key]["mean_fraction"] = (
                        current_fraction + mean_fraction_prev
                    ) / 2
                    Reputation.fraction_changed_history[key]["std_dev_fraction"] = np.sqrt(
                        ((current_fraction - mean_fraction_prev) ** 2 + std_dev_fraction_prev**2) / 2
                    )
                    Reputation.fraction_changed_history[key]["mean_threshold"] = (
                        current_threshold + mean_threshold_prev
                    ) / 2
                    Reputation.fraction_changed_history[key]["std_dev_threshold"] = np.sqrt(
                        ((0.1 * (current_threshold - mean_threshold_prev) ** 2) + std_dev_threshold_prev**2) / 2
                    )

                    data = {
                        "addr": addr,
                        "nei": nei,
                        "round": current_round,
                        "fraction_changed": current_fraction,
                        "threshold": current_threshold,
                        "fraction_score": fraction_score,
                    }
                    Reputation.metrics(scenario, data, addr, nei, "fraction_changed")

                    return max(fraction_score, 0)
                else:
                    return -1
        except Exception:
            logging.exception("Error analyzing anomalies")
            return -1

    @staticmethod
    def save_time_message_history(addr, nei, messages_time_message_normalized, current_round):
        """
        Save the time_message history of a participant (addr) regarding its neighbor (nei) in memory.

        Args:
            addr (str): The identifier of the node whose time_message history is being saved.
            nei (str): The neighboring node involved.
            messages_time_message_normalized (float): The time_message value to be saved.
            current_round (int): The current round number.

        Returns:
            float: The cumulative time_message including the current round.
        """

        try:
            key = (addr, nei)
            avg_time_message = 0

            if key not in Reputation.time_message_history:
                Reputation.time_message_history[key] = {}

            Reputation.time_message_history[key][current_round] = {"time_message": messages_time_message_normalized}

            # logging.info(f"time_message: {messages_time_message_normalized}")
            # logging.info(f"time_message history: {Reputation.time_message_history}")

            # rounds = Reputation.time_message_history[key]
            if messages_time_message_normalized != 0 and current_round >= 4:
                previous_avg = (
                    Reputation.time_message_history[key].get(current_round - 1, {}).get("avg_time_message", None)
                )
                # logging.info(f"Previous avg time_message: {previous_avg}")
                if previous_avg is not None:
                    avg_time_message = (messages_time_message_normalized + previous_avg) / 2
                    # logging.info(f"Avg time_message in if: {avg_time_message}")
                else:
                    avg_time_message = messages_time_message_normalized
                    # logging.info(f"Avg time_message in else: {avg_time_message}")

                Reputation.time_message_history[key][current_round]["avg_time_message"] = avg_time_message
            else:
                avg_time_message = 0

            # logging.info(f"Avg time_message: {avg_time_message}")
            return avg_time_message
        except Exception:
            logging.exception("Error saving time_message history")
            return -1

        except Exception as e:
            logging.exception(f"Error managing model_arrival_latency latency: {e}")
            return 0.0

    @staticmethod
    def manage_model_arrival_latency(round_num, addr, nei, latency, scenario, model_arrival_latency_data):
        """
        Manage the model_arrival_latency latency metric with persistent storage of mean latency.

        Args:
            round_num (int): The round number.
            addr (str): Source IP address.
            nei (str): Destination IP address.
            latency (float): Latency value for the current model_arrival_latency.
            scenario (str): The scenario name for logging and metric storage.
            model_arrival_latency_data (dict): model_arrival_latency-related data.

        Returns:
            float: Normalized model_arrival_latency latency value between 0 and 1.
        """
        try:
            current_key = nei

            if round_num not in Reputation.model_arrival_latency_history:
                Reputation.model_arrival_latency_history[round_num] = {}

            Reputation.model_arrival_latency_history[round_num][current_key] = {
                "latency": latency,
                "score": 0.0,
            }

            # logging.info(f"Reputation.model_arrival_latency_history: {Reputation.model_arrival_latency_history}")

            if round_num >= 5:
                if round_num > 5:
                    if (
                        round_num - 1 in Reputation.model_arrival_latency_history
                        and current_key in Reputation.model_arrival_latency_history[round_num - 1]
                    ):
                        prev_mean_latency = Reputation.model_arrival_latency_history[round_num - 1][current_key].get(
                            "mean_latency", None
                        )
                        prev_stdev_latency = Reputation.model_arrival_latency_history[round_num - 1][current_key].get(
                            "stdev_latency", None
                        )
                        percentil_25 = Reputation.model_arrival_latency_history[round_num - 1][current_key].get(
                            "percentil_25", None
                        )
                        percentil_75 = Reputation.model_arrival_latency_history[round_num - 1][current_key].get(
                            "percentil_75", None
                        )

                        # logging.info(f"Round {round_num} | Node {nei} | Round {round_num - 1} | Previous Mean Latency: {prev_mean_latency} | Previous Stdev Latency: {prev_stdev_latency}")
                        # logging.info(f"Round {round_num} | Node {nei} | Round {round_num - 1} | Percentil 25: {percentil_25} | Percentil 75: {percentil_75}")

                        # Si no se encuentran datos en round_num - 1, recorrer las rondas previas desde round_num - 2 hasta 5
                        if prev_mean_latency is None or prev_stdev_latency is None:
                            for r in range(round_num - 2, 5 - 1, -1):
                                if (
                                    r in Reputation.model_arrival_latency_history
                                    and current_key in Reputation.model_arrival_latency_history[r]
                                ):
                                    prev_mean_latency = Reputation.model_arrival_latency_history[r][current_key].get(
                                        "mean_latency", None
                                    )
                                    prev_stdev_latency = Reputation.model_arrival_latency_history[r][current_key].get(
                                        "stdev_latency", None
                                    )
                                    percentil_25 = Reputation.model_arrival_latency_history[r][current_key].get(
                                        "percentil_25", None
                                    )
                                    percentil_75 = Reputation.model_arrival_latency_history[r][current_key].get(
                                        "percentil_75", None
                                    )

                                    # logging.info(f"Round {round_num} | Node {nei} | Round {r} | Previous Mean Latency: {prev_mean_latency} | Previous Stdev Latency: {prev_stdev_latency}")
                                    # logging.info(f"Round {round_num} | Node {nei} | Round {r} | Percentil 25: {percentil_25} | Percentil 75: {percentil_75}")

                                    # Si se encuentran los datos, salir del bucle
                                    if prev_mean_latency is not None and prev_stdev_latency is not None:
                                        break
                                else:
                                    logging.info(
                                        f"Round {round_num} | Node {nei} | Round {r} | No previous data found."
                                    )

                            # Si no se encontraron datos en ninguna ronda anterior, inicializar valores predeterminados
                            if prev_mean_latency is None or prev_stdev_latency is None:
                                prev_mean_latency = 0
                                prev_stdev_latency = 1
                                percentil_25 = 0
                                percentil_75 = 0
                                # logging.info(f"Round {round_num} | Node {nei} | No previous latency data found, initializing defaults.")
                else:
                    all_latencies = []
                    for r in range(round_num - 1):
                        if r in Reputation.model_arrival_latency_history:
                            for key, data in Reputation.model_arrival_latency_history[r].items():
                                if "latency" in data and data["latency"] != 0:
                                    all_latencies.append(data["latency"])

                    logging.info(f"Round {round_num} | Node {nei} | All latencies: {all_latencies}")

                    prev_mean_latency = np.mean(all_latencies) if all_latencies else 0
                    prev_stdev_latency = np.std(all_latencies) if all_latencies else 1
                    logging.info(
                        f"Round {round_num} | Node {nei} | Initial Mean Latency: {prev_mean_latency} | Initial Stdev Latency: {prev_stdev_latency}"
                    )

                    percentil_25 = np.percentile(all_latencies, 25) if all_latencies else 0
                    percentil_75 = np.percentile(all_latencies, 75) if all_latencies else 0
                    logging.info(
                        f"Round {round_num} | Node {nei} | Percentil 25: {percentil_25} | Percentil 75: {percentil_75}"
                    )

                k = 0.5
                prev_mean_latency = prev_mean_latency + k * (percentil_75 - percentil_25)
                logging.info(f"Round {round_num} | Node {nei} | Mean latency with k: {prev_mean_latency}")
                logging.info(f"Round {round_num} | Node {nei} | Latency: {latency}")

                if latency == 0.0:
                    latency = 0.5

                difference = latency - prev_mean_latency
                logging.info(f"Round {round_num} | Node {nei} | Difference: {difference}")

                if latency <= prev_mean_latency:
                    score = 1
                else:
                    if abs(difference) <= prev_mean_latency:
                        score = 1
                    else:
                        score = 1 / (1 + np.exp(abs(difference) / prev_mean_latency))
                logging.info(f"Round {round_num} | Node {nei} | Score: {score}")
                Reputation.model_arrival_latency_history[round_num][current_key]["score"] = score

                n = round_num if round_num else 1
                # logging.info(f"Round {round_num} | Node {nei} | N: {n}")

                update_mean = (prev_mean_latency * n + latency) / (n + 1)
                update_stdev = np.sqrt(
                    ((prev_stdev_latency**2) + (latency - prev_mean_latency) * (latency - update_mean)) / (n + 1)
                )
                logging.info(
                    f"Round {round_num} | Node {nei} | Mean Latency: {update_mean} | Stdev Latency: {update_stdev}"
                )

                accumulated_latencies = [
                    data["latency"]
                    for r in range(round_num + 1)
                    if r in Reputation.model_arrival_latency_history
                    for key, data in Reputation.model_arrival_latency_history[r].items()
                    if "latency" in data and data["latency"] != 0
                ]
                logging.info(f"Round {round_num} | Node {nei} | Accumulated latencies: {accumulated_latencies}")

                update_percentil_25 = np.percentile(accumulated_latencies, 25) if accumulated_latencies else 0
                update_percentil_75 = np.percentile(accumulated_latencies, 75) if accumulated_latencies else 0
                logging.info(
                    f"Round {round_num} | Node {nei} | Percentil 25: {update_percentil_25} | Percentil 75: {update_percentil_75}"
                )

                Reputation.model_arrival_latency_history[round_num][current_key]["mean_latency"] = update_mean
                Reputation.model_arrival_latency_history[round_num][current_key]["stdev_latency"] = update_stdev
                Reputation.model_arrival_latency_history[round_num][current_key]["percentil_25"] = update_percentil_25
                Reputation.model_arrival_latency_history[round_num][current_key]["percentil_75"] = update_percentil_75

            else:
                # For rounds < 5, no scoring or updates
                score = 0.0

            # Store the metrics
            data = {
                "addr": addr,
                "nei": nei,
                "round": round_num,
                "latency": latency,
                "mean_latency": prev_mean_latency if round_num >= 5 else None,
                "stdev_latency": prev_stdev_latency if round_num >= 5 else None,
                "percentil_25": percentil_25 if round_num >= 5 else None,
                "percentil_75": percentil_75 if round_num >= 5 else None,
                "difference": difference if round_num >= 5 else None,
                "score": score,
            }
            Reputation.metrics(scenario, data, addr, nei, "model_arrival_latency")

            return score

        except Exception as e:
            logging.exception(f"Error managing model_arrival_latency latency: {e}")
            return 0.0

    @staticmethod
    def manage_metric_time_message(messages_time_message, addr, nei, current_round, scenario):
        """
        Manage the time_message metric using percentiles for normalization, considering the last 4 rounds dynamically.

        Args:
            messages_time_message (list): List of messages time_message.
            addr (str): Source IP address.
            nei (str): Destination IP address.
            current_round (int): Current round number.

        Returns:
            float: Normalized time_message value.
            int: Messages count.
        """
        try:
            if current_round == 0:
                return 0.0, 0

            previous_round = current_round - 1
            # logging.info(f"Round {current_round}. Previous round: {previous_round}")

            current_addr_nei = (addr, nei)
            relevant_messages = [
                msg
                for msg in messages_time_message
                if msg["key"] == current_addr_nei and msg["round"] == previous_round
            ]
            messages_count = len(relevant_messages) if relevant_messages else 0
            # logging.info(f"Round {current_round}. Messages count: {messages_count}")

            if previous_round >= 4:
                rounds_to_consider = [previous_round - 4, previous_round - 3, previous_round - 2, previous_round - 1]
            elif previous_round == 3:
                rounds_to_consider = [0, 1, 2]
            elif previous_round == 2:
                rounds_to_consider = [0, 1]
            else:
                rounds_to_consider = list(range(current_round + 1))
            # logging.info(f"Round {current_round}. Rounds to consider: {rounds_to_consider}")

            previous_counts = [
                len([m for m in messages_time_message if m["key"] == current_addr_nei and m["round"] == r])
                for r in rounds_to_consider
            ]
            # logging.info(f"Round {current_round}. Total counts of rounds_to_consider: {previous_counts}")

            # Calculate the 25th and 75th percentiles based on the selected rounds
            Reputation.previous_percentile_25_time_message[current_addr_nei] = (
                np.percentile(previous_counts, 25) if previous_counts else 0
            )
            Reputation.previous_percentile_85_time_message[current_addr_nei] = (
                np.percentile(previous_counts, 85) * 1.15 if previous_counts else 0
            )

            normalized_messages = 0.0

            # Apply the normalizations using the percentiles
            if previous_round >= 4:  # Ensure there's enough data to calculate percentiles
                percentile_25 = Reputation.previous_percentile_25_time_message.get(current_addr_nei, 0)
                # logging.info(f"Round {current_round}. percentile_25: {percentile_25}")
                percentile_85 = Reputation.previous_percentile_85_time_message.get(current_addr_nei, 0)
                # logging.info(f"Round {current_round}. percentile_85: {percentile_85}")

                # logging.info(f"Round {current_round}. Messages count: {messages_count}")
                if percentile_85 - percentile_25 == 0:
                    relative_position = (messages_count - percentile_25) / (percentile_85)
                    # logging.info(f"Round {current_round}. Relative position: {relative_position}")
                else:
                    relative_position = (messages_count - percentile_25) / (percentile_85 - percentile_25)
                    # logging.info(f"Round {current_round}. Relative position: {relative_position}")

                normalized_messages = 1 - 1 / (1 + np.exp(-(relative_position - 1)))
                # logging.info(f"Round {current_round}. Normalized messages sin max: {normalized_messages}")
                normalized_messages = max(0.01, normalized_messages)
                # logging.info(f"Round {current_round}. Normalized messages: {normalized_messages}")

            data = {
                "addr": addr,
                "nei": nei,
                "round": current_round,
                "messages_count": messages_count,
                "normalized_messages": normalized_messages,
                "percentile_25": Reputation.previous_percentile_25_time_message[current_addr_nei],
                "percentile_85": Reputation.previous_percentile_85_time_message[current_addr_nei],
            }
            Reputation.metrics(scenario, data, addr, nei, "time_message")

            return normalized_messages, messages_count
        except Exception:
            logging.exception("Error managing metric time_message")
            return 0.0, 0

    @staticmethod
    def save_model_arrival_latency_history(addr, nei, model_arrival_latency, round_num):
        """
        Save the model_arrival_latency history of a participant (addr) regarding its neighbor (nei) in memory.

        Args:
            addr (str): The identifier of the node whose model_arrival_latency history is being saved.
            nei (str): The neighboring node involved.
            model_arrival_latency (float): The model_arrival_latency value to be saved.
            current_round (int): The current round number.

        Returns:
            float: The cumulative model_arrival_latency including the current round.
        """
        try:
            # Define the current key for the neighbor node (nei)
            current_key = nei

            # Initialize the history for the current round if it doesn't exist
            if round_num not in Reputation.model_arrival_latency_history:
                Reputation.model_arrival_latency_history[round_num] = {}

            # Store the latency value for the current round and neighbor
            if current_key not in Reputation.model_arrival_latency_history[round_num]:
                Reputation.model_arrival_latency_history[round_num][current_key] = {}

            Reputation.model_arrival_latency_history[round_num][current_key].update({
                "score": model_arrival_latency,
            })

            # logging.info(f"model_arrival_latency history: {Reputation.model_arrival_latency_history} | model_arrival_latency: {model_arrival_latency} | current_round: {current_round}")

            if model_arrival_latency > 0 and round_num > 5:
                # logging.info(" if model_arrival_latency >= 0 and round_num > 5")
                previous_avg = (
                    Reputation.model_arrival_latency_history.get(round_num - 1, {})
                    .get(current_key, {})
                    .get("avg_model_arrival_latency", None)
                )
                # logging.info(f"Previous avg model_arrival_latency latency: {previous_avg}")

                if previous_avg is not None:
                    avg_model_arrival_latency = (
                        model_arrival_latency * 0.8 + previous_avg * 0.2
                        if previous_avg is not None
                        else model_arrival_latency
                    )
                    # logging.info(f"Avg model_arrival_latency latency IF: {avg_model_arrival_latency}")
                else:
                    avg_model_arrival_latency = model_arrival_latency - (model_arrival_latency * 0.1)
                    # logging.info(f"Avg model_arrival_latency latency ELSE: {avg_model_arrival_latency}")
            elif model_arrival_latency == 0 and round_num > 5:
                # logging.info(" elif model_arrival_latency == 0 and round_num > 5")
                previous_avg = (
                    Reputation.model_arrival_latency_history.get(round_num - 1, {})
                    .get(current_key, {})
                    .get("avg_model_arrival_latency", None)
                )
                # logging.info(f"Previous avg model_arrival_latency latency: {previous_avg}")
                avg_model_arrival_latency = previous_avg - (previous_avg * 0.05)
                # logging.info(f"Avg model_arrival_latency latency ELIF: {avg_model_arrival_latency}")
            else:
                avg_model_arrival_latency = model_arrival_latency

            Reputation.model_arrival_latency_history[round_num][current_key]["avg_model_arrival_latency"] = (
                avg_model_arrival_latency
            )

            return avg_model_arrival_latency
        except Exception:
            logging.exception("Error saving model_arrival_latency history")

    @staticmethod
    def save_reputation_history_in_memory(addr, nei, reputation, current_round):
        """
        Save the reputation history of a participant (addr) regarding its neighbor (nei) in memory
        and calculate the average reputation.

        Args:
            addr (str): The identifier of the node whose reputation is being saved.
            nei (str): The neighboring node involved.
            reputation (float): The reputation value to be saved.
            current_round (int): The current round number.

        Returns:
            float: The cumulative reputation including the current round.
        """
        try:
            key = (addr, nei)

            if key not in Reputation.reputation_history:
                Reputation.reputation_history[key] = {}

            Reputation.reputation_history[key][current_round] = reputation

            total_reputation = 0
            total_weights = 0
            avg_reputation = 0
            rounds = sorted(Reputation.reputation_history[key].keys(), reverse=True)[:3]

            for i, n_round in enumerate(rounds, start=1):
                rep = Reputation.reputation_history[key][n_round]
                decay_factor = Reputation.calculate_decay_rate(rep) ** i
                total_reputation += rep * decay_factor
                total_weights += decay_factor
                logging.info(f"Round: {n_round}, Reputation: {rep}, Decay: {decay_factor}, Total reputation: {total_reputation}")

            avg_reputation = total_reputation / total_weights
            if total_weights > 0:
                return avg_reputation
            else:
                return -1

        except Exception:
            logging.exception("Error saving reputation history")
            return -1

    @staticmethod
    def calculate_decay_rate(reputation):
        """
        Calculate the decay rate for a reputation value.

        Args:
            reputation (float): Reputation value.

        Returns:
            float: Decay rate.
        """

        if reputation > 0.8:
            return 0.9  # Muy bajo decaimiento
        elif reputation > 0.6:
            return 0.6  # Bajo decaimiento
        elif reputation > 0.4:
            return 0.3  # Alto decaimiento
        else:
            return 0.1  # Muy alto decaimiento

    @staticmethod
    def read_similarity_file(file_path, nei):
        """
        Read a similarity file and extract relevant data for each IP.

        Args:
            file_path (str): Path to the similarity file.

        Returns:
            dict: A dictionary containing relevant data for each IP extracted from the file.
                Each IP will have a dictionary containing cosine, euclidean, minkowski,
                manhattan, pearson_correlation, and jaccard values.
        """
        nei = nei.split(":")[0].strip()
        similarity = 0.0
        with open(file_path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                source_ip = row["source_ip"].split(":")[0].strip()
                if source_ip == nei:
                    try:
                        # Design weights for each similarity metric
                        weight_cosine = 0.4
                        weight_euclidean = 0.2
                        weight_manhattan = 0.2
                        weight_pearson = 0.2

                        # Retrieve and normalize metrics if necessary
                        cosine = float(row["cosine"])
                        euclidean = float(row["euclidean"])
                        manhattan = float(row["manhattan"])
                        pearson_correlation = float(row["pearson_correlation"])

                        # Calculate similarity
                        similarity = (
                            weight_cosine * cosine
                            + weight_euclidean * euclidean
                            + weight_manhattan * manhattan
                            + weight_pearson * pearson_correlation
                        )
                    except Exception:
                        logging.exception("Error reading similarity file")
        return similarity

    @staticmethod
    def metrics(scenario, data, addr, nei, type, update_field=None):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        csv_path = os.path.join(current_dir, f"{scenario}/metrics/{type}/{addr}_{nei}_{type}.csv")
        # logging.info(f"Round {current_round}. CSV path: {csv_path}")
        csv_dir = os.path.dirname(csv_path)

        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        if type != "reputation":
            try:
                with open(csv_path, mode="a", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=data.keys())
                    if file.tell() == 0:
                        writer.writeheader()
                    writer.writerow(data)
            except Exception:
                logging.exception("Error saving messages time_message data to CSV")
        else:
            # logging.info(f"Reputation data received for round {data['round']}: {data}")
            rows = []
            updated = False

            fieldnames = [
                "addr",
                "nei",
                "round",
                "time_message_count",
                "time_message_norm",
                "similarity",
                "fraction",
                "model_arrival_latency",
                "reputation_without_feedback",
                "reputation_with_feedback",
            ]

            if os.path.exists(csv_path):
                with open(csv_path, newline="") as file:
                    rows = list(csv.DictReader(file))
                    # logging.info(f"Existing rows in CSV: {rows}")

                if update_field:
                    for row in rows:
                        if int(row["round"]) == int(data["round"]):
                            # logging.info(f"Updating row for round {data['round']}: {row}")
                            row.update(data)
                            updated = True
                            break

            if not updated:
                rows.append(data)
                # logging.info(f"Appended new data for round {data['round']}: {data}")

            with open(csv_path, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                # logging.info(f"Final rows written to CSV: {rows}")
