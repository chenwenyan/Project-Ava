import time
import os
import json
import re
import numpy as np
from json_repair import repair_json
from typing import Union
from embeddings.BaseEmbeddingModel import BaseEmbeddingModel
from llms.BaseModel import BaseVideoModel, BaseLanguageModel
from .base import (
    BaseGraphStorage,
    BaseVectorStorage,
)
from .prompt import PROMPTS
from .events import extract_events
from .entities import extract_entities_and_relations
from .tree_search import TreeSearch
from video_utils import VideoRepresentation
from bert_score import BERTScorer


def extract_knowledge_graph(
        llm: BaseVideoModel,
        embedding_model: BaseEmbeddingModel,
        knowledge_graph_inst: BaseGraphStorage,
        events_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage,
        relations_vdb: BaseVectorStorage,
        features_vdb: BaseVectorStorage,
        global_config: dict,
        if_check: bool = True
)->Union[BaseGraphStorage]:
    
    if events_vdb.is_empty() or if_check:
        events = extract_events(
            llm=llm,
            video=global_config["video"],
            global_config=global_config,
        )
    else:
        events = events_vdb.get_datas()
    
    if entities_vdb.is_empty() or relations_vdb.is_empty() or if_check:
        entities, relations = extract_entities_and_relations(
            llm=llm,
            embedding_model=embedding_model,
            events=events,
            video=global_config["video"],
            global_config=global_config,
        )
    else:
        entities = entities_vdb.get_datas()
        relations = relations_vdb.get_datas()
    
    # add to event_vdb
    datas_for_vdb = {
        event["id"]:{
            "id": event["id"],
            "description": event["description"],
            "duration": event["duration"],
            "content": [event["description"]]
        } for event in events
    }
    
    events_vdb.upsert(datas_for_vdb)
    
    # add to entities_vdb
    datas_for_vdb = {
        entity["id"]:{
            "id": entity["id"],
            "descriptions": entity["descriptions"],
            "timestamps": entity["timestamps"],
            "frame_indices": entity["frame_indices"],
            "durations": entity["durations"],
            "events": entity["events"],
            "content": entity["descriptions"]
        } for entity in entities
    }
    
    entities_vdb.upsert(datas_for_vdb)
    
    # add to relations_vdb
    datas_for_vdb = {
        relation["id"]:{
            "id": relation["id"],
            "entity1": relation["entity1"],
            "entity2": relation["entity2"],
            "description": relation["description"],
            "content": [relation["description"]]
        } for relation in relations
    }
    
    relations_vdb.upsert(datas_for_vdb)
    
    # add to features vdb
    if features_vdb.is_empty():
        video = global_config["video"]
        datas_for_vdb = {}
        frame_path = video.frames_dir
        for event in events:
            duration = event["duration"]
            for i in range(int(duration[0]), int(duration[1])):
                _, _, index = video.get_frames_by_fps(fps=1, duration=(i,i))
                if not index:
                    continue
                frame_dir = os.path.join(frame_path, f"{index[0]}.jpg")
                datas_for_vdb[f"{i}"] = {
                    "content": [frame_dir],
                    "event": event["id"],
                    "frame_dir": frame_dir
                }
        
        features_vdb.upsert(datas_for_vdb)
    
    for event in events:
        if not knowledge_graph_inst.has_node(event["id"]):
            knowledge_graph_inst.upsert_node(
                node_id=event["id"],
                node_data={
                    "type": "event",
                    "id": event["id"],
                    "description": event["description"],
                }
            )
    
    for entity in entities:
        if not knowledge_graph_inst.has_node(entity["id"]):
            knowledge_graph_inst.upsert_node(
                node_id=entity["id"],
                node_data={
                    "type": "entity",
                    "id": entity["id"],
                    "description": f" ".join(entity["descriptions"]),
                }
            )
        for event_id in entity["events"]:
            if not knowledge_graph_inst.has_edge(entity["id"], event_id):
                knowledge_graph_inst.upsert_edge(
                    entity["id"],
                    event_id,
                    edge_data={
                        "type": "belong_to",
                        "id1": entity["id"],
                        "id2": event_id,
                    }
                )
    
    for relation in relations:
        if not knowledge_graph_inst.has_edge(relation["entity1"], relation["entity2"]):
            knowledge_graph_inst.upsert_edge(
                relation["entity1"],
                relation["entity2"],
                edge_data={
                    "type": "relation",
                    "id1": relation["entity1"],
                    "id2": relation["entity2"],
                    "description": relation["description"]
                }
            )
    
    # add event time relation
    total_events = len(events)
    for i in range(total_events-1):
        if not knowledge_graph_inst.has_edge(events[i]["id"], events[i+1]["id"]):
            knowledge_graph_inst.upsert_edge(
                events[i]["id"],
                events[i+1]["id"],
                edge_data={
                    "type": "event_time_relation",
                    "id1": events[i]["id"],
                    "id2": events[i+1]["id"],
                }
            )
    
    return knowledge_graph_inst

def tree_search(
    query: str,
    llm: BaseLanguageModel,
    video: VideoRepresentation,
    events_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,    
    features_vdb: BaseVectorStorage,
):
    tree_search = TreeSearch(query, llm, video, events_vdb, entities_vdb, features_vdb)
    tree_search.search()
    
    tree_information = tree_search.collect_tree_information()
    return tree_information

def generate_sa_self_consistency_result(
    tree_information: dict,
    llm: BaseLanguageModel,
    self_consistency_num: int = 8,
):
    all_input_prompts = []
    
    for sa_node in tree_information:
        input_prompt = sa_node["input_prompt"]
        for _ in range(self_consistency_num):
            all_input_prompts.append(
                {
                    "text": input_prompt,
                }
            )
    
    # batched generate
    all_responses = llm.batch_generate_response(all_input_prompts)
    
    for i in range(0, len(all_responses), self_consistency_num):
        node_responses = all_responses[i:i+self_consistency_num]
        tree_information[int(i/self_consistency_num)]["responses"] = node_responses
    
    return tree_information

def calculate_sa_score(
    node_results: list,
    scorer: "BERTScorer" = None,
):
    # Allow a pre-built scorer to be reused across many calls (e.g. the
    # progressive early-stopping loop re-scores every round). Building a
    # BERTScorer loads deberta-xlarge-mnli, so we do it once when not supplied.
    if scorer is None:
        scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")
    
    # calculate the score of each node
    for i, node_result in enumerate(node_results):
        responses = node_result["responses"]
        answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        thoughts_counts = {"A": [], "B": [], "C": [], "D": []}
        node_score = {"A": [0.0, 0.0], "B": [0.0, 0.0], "C": [0.0, 0.0], "D": [0.0, 0.0]}
        
        def extract_json(llm_output):
            json_pattern = r'{.*}'
            match = re.search(json_pattern, llm_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    return parsed_json
                except json.JSONDecodeError as e:
                    return f"JSON parse error: {e}"
            else:
                return None
        
        for response in responses:
            json_result = extract_json(response)
            if isinstance(json_result, dict) and "Answer" in json_result.keys() and "Analysis" in json_result.keys():
                answer = json_result["Answer"]
                thoughts = json_result["Analysis"]
                if answer in answer_counts.keys():
                    answer_counts[answer] += 1
                    thoughts_counts[answer].append(thoughts)
        
        self_consistency_num = len(responses)
        
        # cal answer score
        for choice in answer_counts.keys():
            node_score[choice][0] = 1.0 * answer_counts[choice] / self_consistency_num
            
        # cal thoughts score
        for choice in answer_counts.keys():
            if len(thoughts_counts[choice]) > 1:
                text1 = []
                text2 = []
                f1_scores = []
                for i in range(len(thoughts_counts[choice])):
                    for j in range(len(thoughts_counts[choice])):
                        if i != j:
                            text1.append(thoughts_counts[choice][i])
                            text2.append(thoughts_counts[choice][j])
                    f1,_, _ = scorer.score(text1, text2)
                    f1 = f1.mean().item()
                    f1_scores.append(f1)
                node_score[choice][1] = 1.0 * sum(f1_scores) / len(f1_scores)
        
        node_result["scores"] = node_score
    
    return node_results

def choose_best_sa_answer(
    score_results: list,
    alpha: float = 0.3,
):
    for score_result in score_results:
        scores = {
            choice: score_result["scores"][choice][0] * alpha + score_result["scores"][choice][1] * (1 - alpha)
            for choice in score_result["scores"]
        }
        best_choice = max(scores, key=scores.get)
        best_score = scores[best_choice]
        score_result["final_score"] = {best_choice: best_score}

    sorted_score_results = sorted(
        score_results,
        key=lambda x: list(x["final_score"].values())[0],
        reverse=True
    )

    return sorted_score_results


def _sa_round_signature(ranked_nodes: list):
    """Derive the stability signature from ranked SA nodes.

    Returns (signature, diagnostics). `ranked_nodes` is the output of
    choose_best_sa_answer (highest final_score first). The signature is:
        (top1 node path, top1 answer, best-different path|None, best-different answer|None)
    where "best-different" is the highest-scoring node whose final answer
    differs from top1's.
    """
    top1 = ranked_nodes[0]
    top1_answer = list(top1["final_score"].keys())[0]
    top1_score = list(top1["final_score"].values())[0]
    top1_path = top1["path"]

    best_diff = None
    for node in ranked_nodes[1:]:
        answer = list(node["final_score"].keys())[0]
        if answer != top1_answer:
            best_diff = node
            break

    if best_diff is not None:
        bd_answer = list(best_diff["final_score"].keys())[0]
        bd_score = list(best_diff["final_score"].values())[0]
        bd_path = best_diff["path"]
    else:
        bd_answer, bd_score, bd_path = None, None, None

    signature = (
        tuple(top1_path),
        top1_answer,
        tuple(bd_path) if bd_path is not None else None,
        bd_answer,
    )
    diagnostics = {
        "top1_path": top1_path,
        "top1_answer": top1_answer,
        "top1_score": top1_score,
        "best_different_path": bd_path,
        "best_different_answer": bd_answer,
        "best_different_score": bd_score,
    }
    return signature, diagnostics


def generate_sa_self_consistency_result_progressive(
    tree_information: list,
    llm: BaseLanguageModel,
    self_consistency_num: int = 8,
    max_rounds: int = 8,
    stability_rounds: int = 4,
    early_stop: bool = True,
    scorer: "BERTScorer" = None,
):
    """Progressive SA self-consistency with signature-based early stopping.

    Instead of generating `self_consistency_num` responses per node up front
    (num_nodes * self_consistency_num in one batch), this runs round by round:
      * each round generates exactly ONE response per SA node, merged into a
        single batch (reusing generate_sa_self_consistency_result with
        self_consistency_num=1);
      * responses accumulate across rounds per node;
      * after each round the accumulated responses are re-scored with the
        existing calculate_sa_score / choose_best_sa_answer (no scoring logic
        is duplicated here);
      * a stability signature is computed each round. If it is unchanged for
        `stability_rounds` consecutive rounds, generation stops early.

    No node pruning is performed — every node is generated for every executed
    round — so this isolates the effect of early stopping alone.

    Returns (tree_information, meta):
      * tree_information: each node carries the final accumulated `responses`
        (same shape as generate_sa_self_consistency_result), so downstream
        scoring / CA selection is unchanged.
      * meta: per-round diagnostics, executed rounds, total responses, whether
        it stopped early, and the saving ratio vs the fixed full budget.
    """
    if scorer is None:
        scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")

    num_nodes = len(tree_information)
    # Lightweight per-round input; only "input_prompt" is read by the reused fn.
    round_inputs = [{"input_prompt": node["input_prompt"]} for node in tree_information]
    accumulated = [[] for _ in range(num_nodes)]

    prev_signature = None
    stable_count = 0
    rounds_log = []
    executed_rounds = 0
    early_stopped = False

    max_rounds = min(max_rounds, self_consistency_num)

    for round_idx in range(1, max_rounds + 1):
        executed_rounds = round_idx

        # One response per node, merged into a single batch (reuse existing fn).
        round_batch = [{"input_prompt": item["input_prompt"]} for item in round_inputs]
        generate_sa_self_consistency_result(round_batch, llm, self_consistency_num=1)
        for i in range(num_nodes):
            accumulated[i].extend(round_batch[i]["responses"])

        # Re-score accumulated responses with the existing scoring logic.
        scoring_input = [
            {"path": node["path"], "responses": list(accumulated[i])}
            for i, node in enumerate(tree_information)
        ]
        scored = calculate_sa_score(scoring_input, scorer=scorer)
        ranked = choose_best_sa_answer(scored)

        signature, diag = _sa_round_signature(ranked)

        if signature == prev_signature:
            stable_count += 1
        else:
            stable_count = 1
        prev_signature = signature

        responses_so_far = sum(len(r) for r in accumulated)
        rounds_log.append({
            "round": round_idx,
            "responses_so_far": responses_so_far,
            "signature": [
                list(signature[0]),
                signature[1],
                list(signature[2]) if signature[2] is not None else None,
                signature[3],
            ],
            "stable_count": stable_count,
            **diag,
        })

        if early_stop and stable_count >= stability_rounds:
            early_stopped = True
            break

    # Attach final accumulated responses (same shape as the non-progressive fn).
    for i, node in enumerate(tree_information):
        node["responses"] = accumulated[i]

    total_responses = sum(len(r) for r in accumulated)
    full_budget = num_nodes * self_consistency_num
    meta = {
        "num_nodes": num_nodes,
        "self_consistency_num": self_consistency_num,
        "max_rounds": max_rounds,
        "stability_rounds": stability_rounds,
        "early_stop_enabled": early_stop,
        "executed_rounds": executed_rounds,
        "early_stopped": early_stopped,
        "total_responses": total_responses,
        "full_budget_responses": full_budget,
        "saved_responses": full_budget - total_responses,
        "saving_ratio": (full_budget - total_responses) / full_budget if full_budget else 0.0,
        "rounds": rounds_log,
    }
    return tree_information, meta


def generate_ca_self_consistency_result(
    query: str,
    sorted_sa_nodes: list,
    llm: BaseVideoModel,
    video: VideoRepresentation,
    self_consistency_num: int = 4,
    max_frames: int = 256,
    max_retries: int = 3,
):
    candidate_nodes = []
    for sa_node in sorted_sa_nodes:
        if len(candidate_nodes) == 2:
            break
        
        if not candidate_nodes:
            candidate_nodes.append(sa_node)
        else:
            if list(sa_node["final_score"].keys())[0] != list(candidate_nodes[0]["final_score"].keys())[0]:
                candidate_nodes.append(sa_node)
    
    for candidate_node in candidate_nodes:
        frame_durations = candidate_node["frame_durations"]
        
        frames = []
        for frame_duration in frame_durations:
            sampled_frames, _, _ = video.get_frames_by_fps(fps=1, duration=frame_duration)
            frames.extend(sampled_frames)
        
        if len(frames) > max_frames:
            downsampled_indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
            frames = [frames[i] for i in downsampled_indices]
        
        text_prompt = PROMPTS["checkframe_and_answer_COT"].format(
            user_query=query
        )
        
        batch_input_prompts = []
        for _ in range(self_consistency_num):
            batch_input_prompts.append(
                {
                    "text": text_prompt,
                    "video": frames
                }
            )
            
        for _ in range(max_retries):
            try:
                all_responses = llm.batch_generate_response(batch_input_prompts)
                break
            except Exception as e:
                print(e)
                continue
        
        candidate_node["responses"] = all_responses
        
    cleaned_candidate_nodes = [
        {
            "action": "CA",
            "depth": candidate_node["depth"],
            "path": candidate_node["path"],
            "responses": candidate_node["responses"]
        } for candidate_node in candidate_nodes
    ]
    
    return cleaned_candidate_nodes

def calculate_ca_score(
    node_results: list,
):
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")
    
    for node_result in node_results:
        responses = node_result["responses"]
        answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        thoughts_counts = {"A": [], "B": [], "C": [], "D": []}
        node_score = {"A": [0.0, 0.0], "B": [0.0, 0.0], "C": [0.0, 0.0], "D": [0.0, 0.0]}
        
        def extract_json(llm_output):
            json_pattern = r'{.*}'
            match = re.search(json_pattern, llm_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    return parsed_json
                except json.JSONDecodeError as e:
                    return f"JSON parse error: {e}"
            else:
                return None
            
        for response in responses:
            json_result = extract_json(response)
            if isinstance(json_result, dict) and "Answer" in json_result.keys() and "Analysis" in json_result.keys():
                answer = json_result["Answer"]
                thoughts = json_result["Analysis"]
                if answer in answer_counts.keys():
                    answer_counts[answer] += 1
                    thoughts_counts[answer].append(thoughts)
        
        self_consistency_num = len(responses)
        
        for choice in answer_counts.keys():
            node_score[choice][0] = 1.0 * answer_counts[choice] / self_consistency_num
            
        for choice in answer_counts.keys():
            if len(thoughts_counts[choice]) > 1:
                text1 = []
                text2 = []
                f1_scores = []
                for i in range(len(thoughts_counts[choice])):
                    for j in range(len(thoughts_counts[choice])):
                        if i != j:
                            text1.append(thoughts_counts[choice][i])
                            text2.append(thoughts_counts[choice][j])
                    f1,_, _ = scorer.score(text1, text2)
                    f1 = f1.mean().item()
                    f1_scores.append(f1)
                node_score[choice][1] = 1.0 * sum(f1_scores) / len(f1_scores)
        
        node_result["scores"] = node_score
    
    return node_results

def choose_best_ca_answer(
    score_results: list,
    alpha: float = 0.3,
):
    for score_result in score_results:
        scores = {
            choice: score_result["scores"][choice][0] * alpha + score_result["scores"][choice][1] * (1 - alpha)
            for choice in score_result["scores"]
        }
        best_choice = max(scores, key=scores.get)
        best_score = scores[best_choice]
        score_result["final_score"] = {best_choice: best_score}
    
    sorted_score_results = sorted(
        score_results,
        key=lambda x: list(x["final_score"].values())[0],
        reverse=True
    )
    
    return sorted_score_results
                    
            