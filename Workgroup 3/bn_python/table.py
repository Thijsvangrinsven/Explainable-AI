#!/usr/bin/env python
# coding: utf-8

# Imports

# In[5]:


from pyhugin96 import *
from IPython.display import Image
from scipy.special import kl_div, rel_entr
import numpy as np
import math

# Initialize network

# In[134]:


'''
Create a domain object from the endocancer.net file (http://www.cs.ru.nl/~peterl/endomcancer.html)
'''
net_name = "endocancer"
print("Loading network from: ", "{}.net".format(net_name))
endorisk = Domain.parse_domain("{}.net".format(net_name))
endorisk.open_log_file("{}.log".format(net_name))
endorisk.triangulate()
endorisk.compile()
endorisk.close_log_file()

# In[135]:


'''
Check for correct loading of the Domain, if a wrong file is parsed, "NoneType" is outputted
'''
type(endorisk)

# In[136]:


'''
Created an empty dummy node, necessary for the hugin D-sep function, which does not work without all parameters instantiated.
'''
dummy = Node(endorisk)
dummy.set_label("d_sep_dummy")

# In[137]:


'''
Function that resets the domain/network, retracting all evidence and propagating the network.
Params: domain: Domain
'''


def reset(domain):
    if not domain.is_compiled():
        domain.compile()
    domain.retract_findings()
    domain.propagate()


# In[138]:


'''
Function testing the values for the target nodes DSS5 and LNM
Params: domain: Domain
'''


def test_output(domain):
    lnm = domain.get_node_by_name("LNM")
    dss5 = domain.get_node_by_name("Survival5yr")
    print("lnm and dss5 values for yes:")
    return lnm.get_belief(1), dss5.get_belief(0)


# In[139]:


'''Testing reset & test_output functions'''
reset(endorisk)
test_output(endorisk)

# Helper functions

# In[140]:


'''
printerfunction for nodes, prints the label and value of the given node
Params: node: Node
'''


def nprint(node):
    if type(node) == dict:
        for e in node:
            print(e.get_label(), node[e])
    else:
        print(node.get_label())


# In[141]:


'''
remove D-seperated nodes from target
Params: 
domain: Domain
target: Node
evidence: dictionary
'''


def remove_dseparated(domain, target, evidence):
    evidence_copy = dict(evidence)
    ecnodes = list(evidence_copy.keys())
    for e in evidence.keys():
        ecnodes.remove(e)
        dsep = domain.get_d_separated_nodes([target], ecnodes, [dummy])
        # for i in dsep:
        # print(i.get_label())
        # print(e.get_label(), dsep)
        if e in dsep:
            evidence_copy.pop(e)
        ecnodes.append(e)
    return evidence_copy


# In[142]:


'''
Calculates the hellinger distance between two probabilitie lists.
Params: p, q: List
'''


def hellinger_distance_discrete(p, q):
    return (1 / math.sqrt(2)) * math.sqrt(sum(math.pow(math.sqrt(p[i]) - math.sqrt(q[i]), 2) for i in range(len(p))))


# In[143]:


'''
Returns the index and difference for the biggest difference variable of the lists
Params
one, two: List
'''


def max_difference(one, two):
    maxdif = 0
    for i in range(len(one)):
        dif = one[i] - two[i]
        if dif < maxdif:
            maxdif = dif
            index = i
    percentage = round(abs(maxdif) * 100, 1)
    return percentage, index


# In[144]:


'''
Calculates the probability of the target node for 1 or more states, given possible evidence.
Does so by resetting the domain, setting the evidence and running inference on the model, then returning the beliefs for the target node/state
Params: 
domain: Domain
target: Node
evidence: dictionary, default = None
target_state: int, default = None
'''


def P(domain, target, evidence=None, target_state=None):
    reset(domain)
    if evidence != None:
        for i in evidence.keys():
            node = i
            node.select_state(evidence[i])
        domain.propagate()
    result = target
    # return a distribution...
    if target_state != None:
        return target.get_belief(target_state)
    else:
        # print([target.get_belief(0), target.get_belief(1)])
        if target.get_number_of_states() > 2:
            lst = []
            for i in range(target.get_number_of_states()):
                lst.append(target.get_belief(i))
            return lst
        else:
            return [target.get_belief(0), target.get_belief(1)]


# Level 1

# In[145]:


'''
Calculates the impact of an evidence variable e, given all evidence, using the hellinger distance.
Params:
domain: Domain
target: Node
evidence: dictionary
e: dictionary
'''


def impact(domain, target, evidence, e):
    ptE = P(domain, target, evidence)
    evidence_copy = dict(evidence)
    del evidence_copy[e]
    ptEe = P(domain, target, evidence_copy)
    # print(kl_div(ptE, ptEe))
    return hellinger_distance_discrete(ptE, ptEe)


# In[146]:


'''
Determines the threshold theta, as the Hellinger distance between P(T|E) adn G
Params:
domain:Domain
target: Node
evidence: dictionary
alpha: float
'''


def threshold(domain, target, evidence, alpha):
    pte = P(domain, target, evidence)
    G = P(domain, target, evidence) - alpha * (np.array(P(domain, target, evidence)) - P(domain, target))
    return hellinger_distance_discrete(pte, G)


# In[147]:


'''
Determines the significant evidence, and prints alpha
Params
domain: Domain
target: Node
evidence: dictionary
a: list, default: None
'''


def esig(domain, target, evidence, a=None):
    evidence = remove_dseparated(domain, target, evidence)
    esig = {}
    if a == None:
        a = [0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.3, 0.2, 0.1, 0.001]
    c = 0
    keys = list(evidence.keys())
    while len(keys) > len(esig):
        if c < len(a):
            alpha = a[c]
        else:
            print("ran out of alpha's")
            return esig
        theta = threshold(domain, target, evidence, alpha)
        for e in keys:
            if impact(domain, target, evidence, e) > theta:
                # print(e.get_label())
                esig[e] = evidence[e]
                del evidence[e]
                keys.remove(e)
        c += 1
    print("alpha:", alpha)
    return esig


# In[148]:


'''
Calculates the relative risk P(T|E)/P(T|E-e)
Params
domain: Domain
esign: dictionary
target: Node
target_state: int
e: String
'''


def RR(domain, esign, target, target_state, e):
    evidence_copy = dict(esign)
    ptE = P(domain, target, evidence_copy, target_state)
    del evidence_copy[e]
    ptEe = P(domain, target, evidence_copy, target_state)
    return ptE / ptEe


# In[149]:


'''
Determines the direction label of a node e
Params
domain:Domain
esign: dictionary
target: Node
target_state: int
e: String
'''


def direction_label(domain, esign, target, target_state, e):
    evidence_copy = dict(esign)
    if RR(domain, evidence_copy, target, target_state, e) > 1:
        del evidence_copy[e]
        if all(RR(domain, evidence_copy, target, target_state, e1) > 1 for e1 in evidence_copy):
            return "dcons"
        else:
            return "donf"


# In[150]:


def delta_t_e(domain, esign, target, target_state, e):
    evidence_copy = dict(esign)
    ptE = P(domain, target, evidence_copy, target_state)
    del evidence_copy[e]
    ptEe = P(domain, target, evidence_copy, target_state)
    return ptE - ptEe


def delta_t_E(domain, esign, target, target_state, e):
    evidence_copy = dict(esign)
    ptE = P(domain, target, evidence_copy, target_state)
    pt = P(domain, target, target_state=target_state)
    return ptE - pt


def direction_of_change1(domain, esign, target, e):
    if all((delta_t_E(domain, esign, target, t, e) > 0 and delta_t_e(domain, esign, target, t, e) > 0) or (
            delta_t_E(domain, esign, target, t, e) < 0 and delta_t_e(domain, esign, target, t, e) < 0)
           for t in range(target.get_number_of_states())):
        return "dcons"
    elif all((delta_t_E(domain, esign, target, t, e) > 0 and delta_t_e(domain, esign, target, t, e) < 0) or (
            delta_t_E(domain, esign, target, t, e) < 0 and delta_t_e(domain, esign, target, t, e) > 0)
             for t in range(target.get_number_of_states())):
        return "dconf"
    else:
        return "dmix"


def conflict_analysis_direction(domain, esign, target):
    dcons = []
    dconf = []
    dmix = []
    for e in esign.keys():
        if direction_of_change1(domain, esign, target, e) == "dcons":
            dcons.append(e)
        elif direction_of_change1(domain, esign, target, e) == "dconf":
            dconf.append(e)
        elif direction_of_change1(domain, esign, target, e) == "dmix":
            dmix.append(e)
        else:
            print("what the fuck 2")
    return dcons, dconf, dmix


# https://stackoverflow.com/questions/3787908/python-determine-if-all-items-of-a-list-are-the-same-item
def all_same(items):
    return all(x == items[0] for x in items)


def conflict_analysis2(domain, esign, target):
    dcons, dconf, dmix = conflict_analysis_direction(domain, esign, target)
    dominant = []
    consistent = []
    conflicting = dconf
    mixed_consistent = []
    mixed_conflicting = []

    for e in dcons:
        evidence_copy = dict(esign)
        del evidence_copy[e]
        if all(impact(domain, target, esign, e) > impact(domain, target, esign, er) for er in evidence_copy):
            dominant.append(e)
        else:
            consistent.append(e)
    for e1 in dmix:
        count_cons = 0
        count_conf = 0
        for t in range(target.get_number_of_states()):
            if direction_label(domain, esign, target, t, e1) == "dcons":
                count_cons += 1
            elif direction_label(domain, esign, target, t, e1) == "dconf":
                count_conf += 1
        if count_cons > count_conf:
            mixed_consistent.append(e1)
        else:
            mixed_conflicting.append(e1)
    return dominant, consistent, conflicting, mixed_consistent, mixed_conflicting


# Level 2

# In[151]:


'''
Determines the markov blanket of the target, removing the D separated nodes with the significant evidence
Params
domain:Domain
target: Node
evidence: dictionary
esign: dictionary
'''


def markov_blanket(domain, target, evidence, esign):
    XI = []
    for parents in target.get_parents():
        XI.append(parents)
    for children in target.get_children():
        XI.append(children)
        for family in children.get_parents():
            XI.append(family)
    dseperated = remove_dseparated(domain, target, esign)

    XI = list(set(XI))
    for i in dseperated:
        if i in XI:
            XI.remove(i)
    for j in evidence.keys():
        if j in XI:
            XI.remove(j)
    XI.remove(target)
    return XI


# In[152]:


'''
Determines the change of all markov blanket variables after the evidence
'''


def level2(domain, evidence, esign, XI):
    level2_og = {}
    level2_current = {}
    for i in XI:
        level2_og[i] = P(domain, i)
        level2_current[i] = P(domain, i, esign)
    return level2_og, level2_current


# Level 3

# In[153]:


'''
Performs the conflict analysis from level 1 on the markov blanket
Params
domain: Domain
esign: dictionary
XI: List
'''


def level3(domain, esign, XI):
    level3 = {}
    for i in XI:
        evidence = remove_dseparated(domain, i, esign)
        level3[i] = [conflict_analysis2(domain, evidence, i)]
    return level3


# Experimental setup and output

# In[154]:


'''
Table:

parameters:
domain: the bayesian network, hugin domain object
target: the target node, node object
evidence: dictionary with nodes and their instances, e.g. {"LNM": 0 (yes)}
patient: the subject patient, int object
'''


def table(domain, target, evidence, patient):
    esign = esig(domain, target, evidence)
    dominant, consistent, conflicting, mixed_consistent, mixed_conflicting = conflict_analysis2(domain, esign, target)
    XI = markov_blanket(domain, target, evidence, esign)
    lvl2_og, lvl2_new = level2(domain, evidence, esign, XI)
    lvl3 = level3(domain, esign, XI)
    example_case(domain, target, patient)
    print("Level 1")
    print("------------------------------------------------------------------------------")
    # print("The chance of", target.get_label(), "being", target.get_state_label(0) ,"=", target.get_belief(0))
    print("The chance of", target.get_label(), "being", target.get_state_label(1), "=", target.get_belief(1))
    print("------------------------------------------------------------------------------")
    print("What are the factors that support above prediction of", target.get_label() + "?")
    if len(dominant) != 0:
        for i in dominant:
            print(i.get_label(), "=", i.get_state_label(evidence[i]), "(Very important)")
    if len(consistent) != 0:
        for i in consistent:
            print(i.get_label(), "=", i.get_state_label(evidence[i]))
    if len(mixed_consistent) != 0:
        print("Partially supporting:")
        for i in mixed_consistent:
            print(i.get_label(), "=", i.get_state_label(evidence[i]))
    if len(dominant) == 0 and len(consistent) == 0 and len(mixed_consistent) == 0:
        print("None")
    print("------------------------------------------------------------------------------")
    print("What are the factors that do not support above prediction of", target.get_label() + "?")
    if len(conflicting) != 0:
        for i in conflicting:
            print(i.get_label(), "=", i.get_state_label(evidence[i]))
    if len(mixed_conflicting) != 0:
        print("Partially contradicting:")
        for i in mixed_conflicting:
            print(i.get_label(), "=", i.get_state_label(evidence[i]))
    if len(conflicting) == 0 and len(mixed_conflicting) == 0:
        print("None")
    print("------------------------------------------------------------------------------")
    print("Level 2")
    print("------------------------------------------------------------------------------")
    print("How does the model utilize the above factors to predict",
          target.get_label() + "? \nAs the immediate causes of ", target.get_label(), "the model uses:")
    for i in lvl2_og:
        diff, index = max_difference(lvl2_og[i], lvl2_new[i])
        print(i.get_label(), ": ", diff, "% increase in", i.get_state_label(index))
    print("------------------------------------------------------------------------------")
    print("Level 3")
    print("------------------------------------------------------------------------------")
    for node in lvl3:
        print("What are the factors that support above prediction of", node.get_label() + "?")
        if len(lvl3[node][0][0]) != 0:
            for i in lvl3[node][0][0]:
                print(i.get_label(), "=", i.get_state_label(evidence[i]))
        if len(lvl3[node][0][1]) != 0:
            for i in lvl3[node][0][1]:
                print(i.get_label(), "=", i.get_state_label(evidence[i]))
        if len(lvl3[node][0][3]) != 0:
            print("Partially supporting:")
            for i in lvl3[node][0][3]:
                print(i.get_label(), "=", i.get_state_label(evidence[i]))
        if len(lvl3[node][0][0]) == 0 and len(lvl3[node][0][1]) == 0 and len(lvl3[node][0][3]) == 0:
            print("None")
        print("------------------------------------------------------------------------------")
        print("What are the factors that do not support above prediction of", node.get_label() + "?")
        if len(lvl3[node][0][2]) != 0:
            for i in lvl3[node][0][2]:
                print(i.get_label(), "=", i.get_state_label(evidence[i]))
        if len(lvl3[node][0][4]) != 0:
            print("Partially contradicting:")
            for i in lvl3[node][0][4]:
                print(i.get_label(), "=", i.get_state_label(evidence[i]))
        if len(lvl3[node][0][2]) == 0 and len(lvl3[node][0][4]) == 0:
            print("None")
        print("------------------------------------------------------------------------------")

    # In[155]:


'''
Initializes the test patients
Params
endorisk: Domain
target: Node
patient
'''


def example_case(endorisk, target, patient):
    reset(endorisk)
    if patient == 2:
        ca125 = endorisk.get_node_by_name("CA125")
        l1cam = endorisk.get_node_by_name("L1CAM")
        preop = endorisk.get_node_by_name("PrimaryTumor")
        er = endorisk.get_node_by_name("ER")
        pr = endorisk.get_node_by_name("PR")
        p53 = endorisk.get_node_by_name("p53")
        lvsi = endorisk.get_node_by_name("LVSI")
        ctmri = endorisk.get_node_by_name("CTMRI")
        therapy = endorisk.get_node_by_name("Therapy")
        evidence2 = {preop: 2, er: 1, pr: 1, l1cam: 1, p53: 1, lvsi: 1, ca125: 1, ctmri: 1, therapy: 0}
        therapy.select_state(0)
        ca125.select_state(1)
        l1cam.select_state(1)
        preop.select_state(2)
        er.select_state(1)
        pr.select_state(1)
        p53.select_state(1)
        lvsi.select_state(1)
        ctmri.select_state(1)
        endorisk.propagate()
        return evidence2
    elif patient == 1:
        ca125 = endorisk.get_node_by_name("CA125")
        l1cam = endorisk.get_node_by_name("L1CAM")
        preop = endorisk.get_node_by_name("PrimaryTumor")
        atyp = endorisk.get_node_by_name("Cytology")
        evidence = {ca125: 1, l1cam: 1, preop: 1, atyp: 1}
        ca125.select_state(1)
        l1cam.select_state(1)
        preop.select_state(1)
        atyp.select_state(1)
        endorisk.propagate()
        return evidence
    elif patient == 3:
        ca125 = endorisk.get_node_by_name("CA125")
        l1cam = endorisk.get_node_by_name("L1CAM")
        preop = endorisk.get_node_by_name("PrimaryTumor")
        atyp = endorisk.get_node_by_name("Cytology")
        pr = endorisk.get_node_by_name("PR")
        er = endorisk.get_node_by_name("ER")
        evidence = {ca125: 0, l1cam: 0, preop: 0, atyp: 0, pr: 0, er: 0}
        ca125.select_state(0)
        l1cam.select_state(0)
        preop.select_state(0)
        atyp.select_state(0)
        pr.select_state(0)
        er.select_state(0)
        endorisk.propagate()
        return evidence
    else:
        print("unknown patient")


# In[156]:


'''
Sets the target, creates the evidence and runs the table method for patient 2
'''
target = endorisk.get_node_by_name("LNM")
evidence = example_case(endorisk, target, 3)
table(endorisk, target, evidence, 3)

