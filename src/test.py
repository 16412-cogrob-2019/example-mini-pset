import networkx as nx
import numpy as np
from nose.tools import assert_equal, ok_
from IPython.display import display, HTML, clear_output

def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print("Tests passed!!") 


def test_expand_gadget(fn):  
    Time_Expanded_G = nx.DiGraph()
    
    CUG = nx.DiGraph()
    CUG.add_node("A")
    CUG.add_node("B")
    CUG.add_edge("A", "B")

    Goal_Nodes = []
    t = 0
    
    fn(Time_Expanded_G, CUG, Goal_Nodes, t)
    
    answer = nx.DiGraph()      
    answer.add_edge("A_0_p", "A_B_0_p", weight=0, capacity=1)
    answer.add_edge("B_0_p", "A_B_0_p", weight=0, capacity=1)
    answer.add_edge("A_B_0_p", "A_B_1", weight=1, capacity=1)
    answer.add_edge("A_B_1", "A_1", weight=0, capacity=1)
    answer.add_edge("A_B_1", "B_1", weight=0, capacity=1)
    
    if set(Time_Expanded_G.nodes()) != set(answer.nodes()):
        raise Exception("Expanded gadget has wrong nodes!")
    if len(Time_Expanded_G.edges()) != len(answer.edges()):
        raise Exception("Expanded gadget has wrong number of edges!")
    for (u, v) in answer.edges():
        weight = answer[u][v]['weight']
        if Time_Expanded_G[u][v]['weight'] != weight:
            raise Exception("Invalid weight!")
        capacity = answer[u][v]['capacity']
        if Time_Expanded_G[u][v]['capacity'] != capacity:
            raise Exception("Invalid capacity!")         
    return True  

def create_expand_parallel_answer(graph):
    graph.add_node("A_2_p")
    graph.add_node("B_2_p")
    graph.add_node("C_2_p")
    
    graph.add_edge("A_0_p", "A_B_0_p", weight=0, capacity=1)
    graph.add_edge("B_0_p", "A_B_0_p", weight=0, capacity=1)
    graph.add_edge("A_B_0_p", "A_B_1", weight=1, capacity=1)
    graph.add_edge("A_B_1", "A_1", weight=0, capacity=1)
    graph.add_edge("A_B_1", "B_1", weight=0, capacity=1)
    
    graph.add_edge("B_0_p", "B_C_0_p", weight=0, capacity=1)
    graph.add_edge("C_0_p", "B_C_0_p", weight=0, capacity=1)
    graph.add_edge("B_C_0_p", "B_C_1", weight=1, capacity=1)
    graph.add_edge("B_C_1", "B_1", weight=0, capacity=1)
    graph.add_edge("B_C_1", "C_1", weight=0, capacity=1)      
    
    graph.add_edge("A_1_p", "A_B_1_p", weight=0, capacity=1)
    graph.add_edge("B_1_p", "A_B_1_p", weight=0, capacity=1)
    graph.add_edge("A_B_1_p", "A_B_2", weight=1, capacity=1)
    graph.add_edge("A_B_2", "A_2", weight=0, capacity=1)
    graph.add_edge("A_B_2", "B_2", weight=0, capacity=1)

    graph.add_edge("B_1_p", "B_C_1_p", weight=0, capacity=1)
    graph.add_edge("C_1_p", "B_C_1_p", weight=0, capacity=1)
    graph.add_edge("B_C_1_p", "B_C_2", weight=1, capacity=1)
    graph.add_edge("B_C_2", "B_2", weight=0, capacity=1)
    graph.add_edge("B_C_2", "C_2", weight=0, capacity=1)    
    
    graph.add_edge("A_0_p", "A_1", weight=1, capacity=1) 
    graph.add_edge("B_0_p", "B_1", weight=1, capacity=1) 
    graph.add_edge("C_0_p", "C_1", weight=0, capacity=1)  
    graph.add_edge("A_1_p", "A_2", weight=1, capacity=1) 
    graph.add_edge("B_1_p", "B_2", weight=1, capacity=1)
    graph.add_edge("C_1_p", "C_2", weight=0, capacity=1)  
    graph.add_edge("A_1", "A_1_p", weight=0, capacity=1)  
    graph.add_edge("B_1", "B_1_p", weight=0, capacity=1)  
    graph.add_edge("C_1", "C_1_p", weight=0, capacity=1)
    graph.add_edge("A_2", "A_2_p", weight=0, capacity=1)  
    graph.add_edge("B_2", "B_2_p", weight=0, capacity=1)  
    graph.add_edge("C_2", "C_2_p", weight=0, capacity=1)      

def test_expand_parallel(fn):      
    CUG = nx.DiGraph()
    CUG.add_node("A")
    CUG.add_node("B")
    CUG.add_node("C")
    CUG.add_edge("A", "B")
    CUG.add_edge("B", "C")

    Goal_Nodes = ["C"]
    t = 0
    
    ###### Time_Expanded_G #####
    
    Time_Expanded_G = nx.DiGraph()
    
    Time_Expanded_G.add_edge("A_0_p", "A_B_0_p", weight=0, capacity=1)
    Time_Expanded_G.add_edge("B_0_p", "A_B_0_p", weight=0, capacity=1)
    Time_Expanded_G.add_edge("A_B_0_p", "A_B_1", weight=1, capacity=1)
    Time_Expanded_G.add_edge("A_B_1", "A_1", weight=0, capacity=1)
    Time_Expanded_G.add_edge("A_B_1", "B_1", weight=0, capacity=1)
    
    Time_Expanded_G.add_edge("B_0_p", "B_C_0_p", weight=0, capacity=1)
    Time_Expanded_G.add_edge("C_0_p", "B_C_0_p", weight=0, capacity=1)
    Time_Expanded_G.add_edge("B_C_0_p", "B_C_1", weight=1, capacity=1)
    Time_Expanded_G.add_edge("B_C_1", "B_1", weight=0, capacity=1)
    Time_Expanded_G.add_edge("B_C_1", "C_1", weight=0, capacity=1)       
    
    fn(Time_Expanded_G, CUG, Goal_Nodes, t)
    
    Time_Expanded_G.add_edge("A_1_p", "A_B_1_p", weight=0, capacity=1)
    Time_Expanded_G.add_edge("B_1_p", "A_B_1_p", weight=0, capacity=1)
    Time_Expanded_G.add_edge("A_B_1_p", "A_B_2", weight=1, capacity=1)
    Time_Expanded_G.add_edge("A_B_2", "A_2", weight=0, capacity=1)
    Time_Expanded_G.add_edge("A_B_2", "B_2", weight=0, capacity=1)

    Time_Expanded_G.add_edge("B_1_p", "B_C_1_p", weight=0, capacity=1)
    Time_Expanded_G.add_edge("C_1_p", "B_C_1_p", weight=0, capacity=1)
    Time_Expanded_G.add_edge("B_C_1_p", "B_C_2", weight=1, capacity=1)
    Time_Expanded_G.add_edge("B_C_2", "B_2", weight=0, capacity=1)
    Time_Expanded_G.add_edge("B_C_2", "C_2", weight=0, capacity=1)     
    
    fn(Time_Expanded_G, CUG, Goal_Nodes, t+1)
    
    ###### answer #####
    
    answer = nx.DiGraph()
    create_expand_parallel_answer(answer)
       
    if set(Time_Expanded_G.nodes()) != set(answer.nodes()):
        raise Exception("Expanded parallel has wrong nodes!")
    if len(Time_Expanded_G.edges()) != len(answer.edges()):
        raise Exception("Expanded parallel has wrong number of edges!")
    for (u, v) in answer.edges():
        weight = answer[u][v]['weight']
        if Time_Expanded_G[u][v]['weight'] != weight:
            raise Exception("Expanded parallel has invalid weight!")
        capacity = answer[u][v]['capacity']
        if Time_Expanded_G[u][v]['capacity'] != capacity:
            raise Exception("Expanded parallel has invalid capacity!")         
    return True 

def test_set_terminals(fn):    
    CUG = nx.DiGraph()
    CUG.add_node("A")
    CUG.add_node("B")
    CUG.add_node("C")
    CUG.add_edge("A", "B")
    CUG.add_edge("B", "C")

    Initial_Nodes = ["A"]
    Goal_Nodes = ["C"]   
    t=0
    
    ###### Time_Expanded_G #####

    Time_Expanded_G = nx.DiGraph()
    create_expand_parallel_answer(Time_Expanded_G)    

    T=2
    fn(Time_Expanded_G, Initial_Nodes, Goal_Nodes,T)

    ###### answer #####
    
    answer = nx.DiGraph()
    create_expand_parallel_answer(answer)     
  
    answer.add_node("A_0_p", demand=-1)
    answer.add_node("C_2", demand=1)
    
    if set(Time_Expanded_G.nodes()) != set(answer.nodes()):
        raise Exception("Set terminals has wrong nodes!")
    if len(Time_Expanded_G.edges()) != len(answer.edges()):
        raise Exception("Set terminals has wrong number of edges!")
    Time_Expanded_G_demands = nx.get_node_attributes(Time_Expanded_G,'demand')   
    answer_demands = nx.get_node_attributes(answer,'demand')    
    initialDemand =  answer_demands['A_0_p']    
    if initialDemand != Time_Expanded_G_demands['A_0_p']: 
        raise Exception("Set terminals has invalid initial demand!")   
    goalDemand =  answer_demands['C_2']    
    if goalDemand != Time_Expanded_G_demands['C_2']:
        raise Exception("Set terminals has invalid goal demand!")      
    return True  

def test_expand_CUG(fn):
    CUG = nx.DiGraph()
    CUG.add_node("A")
    CUG.add_node("B")
    CUG.add_node("C")
    CUG.add_edge("A", "B")
    CUG.add_edge("B", "C")

    Initial_Nodes = ["A"]
    Goal_Nodes = ["C"]   
    T=2    
    
    Time_Expanded_G = fn(CUG, T, Initial_Nodes, Goal_Nodes)
    
    ###### answer #####
    
    answer = nx.DiGraph()
    create_expand_parallel_answer(answer)     
  
    answer.add_node("A_0_p", demand=-1)
    answer.add_node("C_2", demand=1)
    
    if set(Time_Expanded_G.nodes()) != set(answer.nodes()):
        raise Exception("Expanded CUG has wrong nodes!")
    if len(Time_Expanded_G.edges()) != len(answer.edges()):
        raise Exception("Expanded CUG has wrong number of edges!")
    for (u, v) in answer.edges():
        weight = answer[u][v]['weight']
        if Time_Expanded_G[u][v]['weight'] != weight:
            raise Exception("Expanded CUG has invalid weight!")
        capacity = answer[u][v]['capacity']
        if Time_Expanded_G[u][v]['capacity'] != capacity:
            raise Exception("Expanded CUG has invalid capacity!")              
    Time_Expanded_G_demands = nx.get_node_attributes(Time_Expanded_G,'demand')   
    answer_demands = nx.get_node_attributes(answer,'demand')      
    initialDemand =  answer_demands['A_0_p']    
    if initialDemand != Time_Expanded_G_demands['A_0_p']: 
        raise Exception("Expanded CUG has invalid initial demand!")   
    goalDemand =  answer_demands['C_2']    
    if goalDemand != Time_Expanded_G_demands['C_2']:
        raise Exception("Expanded CUG has invalid goal demand!")   
    return True