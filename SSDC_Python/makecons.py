import numpy as np

def MakeCons(gnd, N=0.10, seed = 42):
    """Función para crear restricciones.

    Args:
        gnd (list): Lista de etiquetas. Un valor -1 o "-1" indica un dato no etiquetado.
        N (int): Porcentaje de restricciones.

    Returns:
        tuple: Una tupla que contiene dos matrices de restricciones CL y ML.

    """
    ND = np.argwhere(np.logical_and(gnd != -1, gnd != "-1"))

    # Calculamos el número total de pares únicos que podemos formar con ND puntos
    total_pairs = len(ND) * (len(ND) - 1) // 2

    # Inicializamos una matriz vacía para los pares de restricciones.
    # La matriz tiene 'total_pairs' filas y 2 columnas para almacenar pares de índices.
    constraints = np.zeros((total_pairs, 2), dtype=int)
    # 'x' es el índice para empezar a llenar la matriz de restricciones en cada iteración
    k=0
    for indi, i in enumerate(ND[:-1]):
        for j in ND[indi+1:]:
            constraints[k,0], constraints[k,1] = i[0], j[0]
            k = k+1

    # Seleccionamos aleatoriamente 'fix(ND * (N / 10))' pares de restricciones
    num_constraints = int(total_pairs * (N))
    np.random.seed(seed)
    # print(np.random.permutation(ND))
    # print(len(ND))
    selected_indices = np.random.permutation(len(constraints))[:num_constraints]
    # print(selected_indices)
    # print(constraints)
    # Inicializamos las listas de restricciones Cannot-Link (CL) y Must-Link (ML)
    CL = []
    ML = []

    # Evaluamos los pares seleccionados y los clasificamos en CL o ML
    for idx in selected_indices:
        # Si las etiquetas en 'gnd' para el par actual son diferentes, agregamos a CL
        if gnd[constraints[idx, 0]] != gnd[constraints[idx, 1]]:
            # print(idx)
            # print(constraints[idx-2:idx+2])
            CL.append(constraints[idx])
        # Si las etiquetas son iguales, agregamos a ML
        else:
            ML.append(constraints[idx])

    # Convertimos las listas de CL y ML a arrays de NumPy para facilitar su manipulación posterior
    CL = np.array(CL)
    ML = np.array(ML)

    # Devolvemos las matrices de restricciones CL y ML
    return CL, ML