import re
import javalang
import numpy as np
import os

"""
Splits camelCase, PascalCase, and snake_case into separate words

:param string: the string that needs to be split into separate words
:returns: a list of strings where each string is a single word

Example:
  Input:
    'camelCase'
  Output:
    ['camel', 'Case']

Example 2:
  Input:
    'PascalXYCase'
  Output:
    ['Pascal', 'XY', 'Case']
"""
def split_case(string):
  return re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', string)

""" 
Flatten a list

:param lst: the list to flatten
:returns: the flattened list

Example:
  Input:
  [ 
    [
      [
        'variable_1_part_1', 
        ..., 
        'variable_1_part_m'
      ],
      ...,
      [
        'variable_k_part_1', 
        ...,
        'variable_k_part_n'
      ]
    ], 
    [
      [
        'method_1_part_1', 
        ...,
        'method_1_part_o'
      ], 
      ..., 
      [
        'method_l_part_1', 
        ..., 
        'method_l_part_p'
      ]
    ] 
  ]

  Output:
  [
    'variable_1_part_1', ..., 'variable_1_part_m',
    ..., 
    'variable_k_part_1', ..., 'variable_k_part_n',
    'method_1_part_1', ..., 'method_1_part_o', 
    ...,
    'method_l_part_1', ..., 'method_l_part_p'
  ]
"""
def flatten(lst):
  result = []
  for item in lst:
    if isinstance(item, list):  # Check if the item is a list
      result.extend(flatten(item))  # Recursively flatten it
    else:
      result.append(item)  # Append non-list items
  return result

"""
Calculate the projection length with regard to the projection axis

:param v1: the projected vector
:param v2: the vector that the projected vector is projected onto
:returns: the amount the `v1` points into the direction of `v2`

Example:
  Input:
    v1 = numpy.array([12,20])
    v2 = numpy.array([4,0])
  Output: 
    12
"""
def calculate_projection_length(v1, v2):
  return np.dot(v1, v2) / (np.linalg.norm(v2))

"""
Find the embedding of a package given its path

:param package_path: the path to the directory of the package of which the embedding needs to be determined
:param model: the fasttext model to use for retrieving word embeddings
:returns: the embedding of the package or None if no embedding exists (package does not contain .java files)
"""
def get_package_embedding(package_path, model):
  files = next(os.walk(package_path))[2]
  files = list(filter(lambda file : file.endswith(".java"), files))
  class_embeddings = []
  for file in files:
    class_path = package_path + "/" + file
    class_embeddings.append(get_class_embedding(class_path, model))
  return np.mean(class_embeddings, axis=0) if len(class_embeddings) > 0 else None

"""
Find the embedding of a class given its path

:param class_path: the file path to the class of which the embedding needs to be determined
:param model: the fasttext model to use for retrieving word embeddings
:returns: the embedding of the class
"""
def get_class_embedding(class_path, model):
  field_identifiers, method_identifiers = get_class_identifiers(class_path)
 
  split_file_name = list(map(lambda x : x.lower(), split_case(os.path.basename(class_path).removesuffix(".java"))))
  context = set(flatten([field_identifiers, method_identifiers]))
  context_embedding = np.mean([model.get_word_vector(word) for word in context], axis=0) if len(context) > 0 else None
  class_name_embedding = np.mean([model.get_word_vector(word) for word in split_file_name], axis=0)
  class_embedding = class_name_embedding if context_embedding is None else np.mean([class_name_embedding, context_embedding], axis=0)
  
  return class_embedding

"""
Find the field identifiers and method identifiers of a class given the file path to the class

:param class_path: the file path to the class of which the identifiers need to be determined
:returns: a tuple `(field_identifiers, method_identifiers)` 
          where `field_identifiers` is a list of field identifiers `[field_identifier1, field_identifier2, ...]` 
          where each `field_identifier` is a list of its parts in lowercase (e.g. variableName is turned into ['variable', 'name']). 
          The same holds for method_identifiers, but the first part of each identifier is removed (e.g. getTotalCount is turned into ['total', 'count'])
"""
def get_class_identifiers(class_path):
  class_file = open(class_path)
  src_text = "".join(class_file.readlines())
  src_tree = javalang.parse.parse(src_text)
  class_file.close()

  # Collect all field identifiers and method identifiers in raw format (e.g. fieldIdentifier, methodIdentifier)
  field_identifiers = []
  method_identifiers = []

  if len(src_tree.types) == 0:
    return ([], [])

  for declaration in src_tree.types[0].body:
    if isinstance(declaration, javalang.tree.MethodDeclaration):
      method_identifiers.append(declaration.name)
    elif isinstance(declaration, javalang.tree.FieldDeclaration):
      for variable in declaration.declarators:
        field_identifiers.append(variable.name)

  # Post-process the identifiers (split camelCase, PascalCase and snake_case, and lowercase the words)
  field_identifiers = list(map(lambda str : split_case(str), field_identifiers))
  for i in range(len(field_identifiers)):
    field_identifiers[i] = list(map(lambda str : str.lower(), field_identifiers[i]))
  method_identifiers = list(map(lambda str : split_case(str), method_identifiers))
  for i in range(len(method_identifiers)):
    method_identifiers[i].remove(method_identifiers[i][0])
  method_identifiers = [x for x in method_identifiers if len(x) != 0]
  for i in range(len(method_identifiers)):
    method_identifiers[i] = list(map(lambda str : str.lower(), method_identifiers[i]))
  
  return (field_identifiers, method_identifiers)