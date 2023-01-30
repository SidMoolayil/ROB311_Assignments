from collections import deque
from support import definite_clause

### THIS IS THE TEMPLATE FILE
### WARNING: DO NOT CHANGE THE NAME OF FILE OR THE FUNCTION SIGNATURE


def pl_fc_entails(symbols_list : list, KB_clauses : list, known_symbols : list, query : int) -> bool:
    """
    pl_fc_entails function executes the Propositional Logic forward chaining algorithm (AIMA pg 258).
    It verifies whether the Knowledge Base (KB) entails the query
        Inputs
        ---------
            symbols_list  - a list of symbol(s) (have to be integers) used for this inference problem
            KB_clauses    - a list of definite_clause(s) composed using the numbers present in symbols_list
            known_symbols - a list of symbol(s) from the symbols_list that are known to be true in the KB (facts)
            query         - a single symbol that needs to be inferred

            Note: Definitely check out the test below. It will clarify a lot of your questions.

        Outputs
        ---------
        return - boolean value indicating whether KB entails the query
    """
    ### START: Your code
    # initialization
    clause_with_symbol = {}
    count = []

    # set up symbol to clauses dictionary
    for s in symbols_list:
        clause_with_symbol[s] = []

    for i in range(len(KB_clauses)):
        # store counts
        count.append(len(KB_clauses[i].body))
        for j in KB_clauses[i].body:
            # store clauses
            clause_with_symbol[j].append(i)

    agenda = known_symbols
    # loop until agenda is empty
    while agenda:
        p = agenda.pop()
        # if query in KB, known to be True
        if p == query:
            return True
        # if p is part of a KB clause, loop over all clauses p is part of
        for i in clause_with_symbol[p]:
            # '.' symbols list distinct, decrement counts of symbols required to infer conclusion
            count[i] -= 1
            # if all symbols are part of KB, conclusion can be inferred and added to KB
            if count[i] == 0:
                agenda.append(KB_clauses[i].conclusion)
    return False # remove line if needed
    ### END: Your code


# SAMPLE TEST
if __name__ == '__main__':

    # Symbols used in this inference problem (Has to be Integers)
    symbols = [1,2,9,4,5]

    # Clause a: 1 and 2 => 9
    # Clause b: 9 and 4 => 5
    # Clause c: 1 => 4
    KB = [definite_clause([1, 2], 9), definite_clause([9,4], 5), definite_clause([1], 4)]

    # Known Symbols 1, 2
    known_symbols = [2, 1]

    # Does KB entail 5?
    entails = pl_fc_entails(symbols, KB, known_symbols, 5)

    print("Sample Test: " + ("Passed" if entails == True else "Failed"))
