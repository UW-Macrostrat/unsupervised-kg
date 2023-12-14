import os
from neo4j import GraphDatabase
import argparse
import pandas as pd

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--login_file', type= str, required = True, help = "The path to the file containing the login info")
    parser.add_argument('--graph_file', type= str, required = True, help = "The path to the csv file containing the graph metadata")
    return parser.parse_args()

def read_neo4j_config(info_file):
    config = {}
    with open(info_file, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if len(stripped_line) == 0 or " " in stripped_line or "=" not in stripped_line:
                continue

            key, value = stripped_line.split('=')
            config[key] = value

    return config

def neo_create_node(tx, node_name):
    query = (
        f"MERGE (n:Node {{name: '{node_name}'}})"
    )
    tx.run(query)

def create_nodes_and_relationships(tx, src, rel_type, dst, article_id, sentence):
    escaped_sentence = sentence.replace("'", "''")
    query = (
        f"MERGE (src:Node {{name: '{src}'}}) "
        f"MERGE (dst:Node {{name: '{dst}'}}) "
        f"MERGE (src)-[r:`{rel_type}`]->(dst) "
        f"SET r.article_id = '{article_id}', r.sentence = '{escaped_sentence}' "
    )
    tx.run(query)

def create_all_nodes(graph_df, driver):
    # Get all of the nodes
    src_nodes = set(graph_df["src"].values)
    dst_nodes = set(graph_df["dst"].values)
    all_nodes = src_nodes.union(dst_nodes)
    print("Ensuring we have all", len(all_nodes), "nodes")

    # Add all of the nodes
    with driver.session() as session:
        for curr_node in all_nodes:
            session.execute_write(neo_create_node, curr_node)

def create_all_relationships(graph_df, driver):
    with driver.session() as session:
        print("Creating the", len(graph_df.index), "relationships")
        for index, row in graph_df.iterrows():
            session.execute_write(create_nodes_and_relationships, row["src"], row["type"], row["dst"], row["article_id"], row["sentence"])

def main():
    args = read_args()
    config = read_neo4j_config(args.login_file)
    graph_df = pd.read_csv(args.graph_file)
    
    with GraphDatabase.driver(config.get('NEO4J_URI'), auth=(config.get('NEO4J_USERNAME'), config.get('NEO4J_PASSWORD'))) as driver:
        create_all_nodes(graph_df, driver)
        create_all_relationships(graph_df, driver)


if __name__ == "__main__":
    main()