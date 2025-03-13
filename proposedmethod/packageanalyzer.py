import sys
import os
import numpy as np
from enum import Enum
from common import get_package_embedding, calculate_projection_length

class PackageType(Enum):
  CLASSES_AND_CLASS_PACKAGES = 0
  NO_CLASSES_BUT_CLASS_PACKAGES = 1
  CLASSES_BUT_NO_CLASS_PACKAGES = 2
  NO_CLASSES_AND_NO_CLASS_PACKAGES = 3

"""
Check whether a package contains classes

:param this_package_path: the path to the package
:returns: whether the package contains classes given its path
"""
def contains_classes(this_package_path):
  files = next(os.walk(this_package_path))[2]
  files = list(filter(lambda file : file.endswith(".java"), files))
  return len(files) > 0

"""
Check whether a package contains subpackages containing classes

:param this_package_path: the path to the package
:returns: whether the package contains a subpackage that contains classes
"""
def contains_class_packages(this_package_path):
  subpackages = next(os.walk(this_package_path))[1]
  for subpackage in subpackages:
    subpackage_path = this_package_path + "/" + subpackage
    if contains_classes(subpackage_path):
      return True
  return False

"""
Get all subpackages organized into a list of ones that contain classes and a list of ones that do not

:param this_package_path: the path to the package in which to look for subpackages
:returns: a tuple of 2 lists, the first being a list of paths to packages that contain classes, and the second being a list of paths to packages that do not contain classes
"""
def get_subpackages(this_package_path):
  class_package_paths = []
  subdiv_package_paths = []
  subpackages = next(os.walk(this_package_path))[1]
  for subpackage in subpackages:
    subpackage_path = this_package_path + "/" + subpackage
    if contains_classes(subpackage_path):
      class_package_paths.append(subpackage_path)
    else:
      subdiv_package_paths.append(subpackage_path)
  return class_package_paths, subdiv_package_paths

"""
Get a package's type

:param this_package_path: the path to the package under consideration
:returns: the package's type (whether it contains classes and whether it contains subpackages that contain classes)
"""
def get_package_type(this_package_path):
  if (contains_classes(this_package_path) and contains_class_packages(this_package_path)):
    return PackageType.CLASSES_AND_CLASS_PACKAGES
  elif not contains_classes(this_package_path) and contains_class_packages(this_package_path):
    return PackageType.NO_CLASSES_BUT_CLASS_PACKAGES
  elif contains_classes(this_package_path) and not contains_class_packages(this_package_path):
    return PackageType.CLASSES_BUT_NO_CLASS_PACKAGES
  else:
    return PackageType.NO_CLASSES_AND_NO_CLASS_PACKAGES

"""
Adds the projections of all class package embeeddings onto the embedding of their parent package (this_package_embedding) to the list of projections

:param class_package_paths: the list of paths to packages containing classes
:param this_package_embedding: the embedding of the package under consideration (the one that is the parent of all class_package_paths)
:param projections: the list to add projections to
:param model: the fasttext model to use for retrieving word embeddings
"""
def add_class_package_on_parent_package_projections(class_package_paths, this_package_embedding, projections, model):
  # Projection of all ClassPackages onto ThisPackage
  for class_package_path in class_package_paths:
    projections.append(calculate_projection_length(get_package_embedding(class_package_path, model), this_package_embedding))

"""
Adds the projections of all class package embeddings onto each other to the list of projections

:param class_package_paths: the list of paths to packages containing classes
:param this_package_embedding: the embedding of the package under consideration (the one that is the parent of all class_package_paths and all subdiv_package_paths)
:param projections: the list to add projections to
:param is_root_package: whether the package under consideration is the directory provided by the user
:param model: the fasttext model to use for retrieving word embeddings
"""
def add_class_package_pairwise_projections(class_package_paths, projections, is_root_package, model):
  # Projection of all ClassPackages in ThisPackage onto each other, but not if ThisPackage is the root package
  if is_root_package:
    return

  for class_package_path_a in class_package_paths:
    for class_package_path_b in class_package_paths:
      if class_package_path_a != class_package_path_b:
        projections.append(calculate_projection_length(get_package_embedding(class_package_path_a, model), get_package_embedding(class_package_path_b, model)))

"""
Adds the projections of all subdiv package embeddings onto their parent package embedding (this_package_embedding). A subdiv package's embedding exists if and only if
the subdiv package contains class packages. Then the embedding is the average of all projections of embeddings of class packages onto the parent of the subdiv package
(this_package_embedding).

:param subdiv_package_paths: the list of paths to packages not containing classes
:param this_package_embedding: the embedding of the package under consideration (the one that is the parent of all class_package_paths and all subdiv_package_paths)
:param projections: the list to add projections to
:param model: the fasttext model to use for retrieving word embeddings
"""
def add_subdiv_package_on_parent_package_projections(subdiv_package_paths, this_package_embedding, projections, model):
  # Projection of all SubdivPackages p in ThisPackage onto ThisPackage provided that p contains class packages.
  for subdiv_package_path in subdiv_package_paths:
    if contains_class_packages(subdiv_package_path):
      class_package_in_subdiv_paths, _ = get_subpackages(subdiv_package_path)
      class_package_in_subdiv_embeddings = [get_package_embedding(class_package_in_subdiv_path, model) for class_package_in_subdiv_path in class_package_in_subdiv_paths]
      subdiv_package_embedding = np.mean(class_package_in_subdiv_embeddings, axis=0)
      projections.append(calculate_projection_length(subdiv_package_embedding, this_package_embedding))

"""
Add projections to the list of projections recursively given a root package

:param this_package_path: the path to the root package
:param projections: the list to add projections to
:param model: the fasttext model to use for retrieving word embeddings
:param is_root_package: whether the package under consideration is the directory provided by the user
"""
def find_all_projections_recursively(this_package_path, projections, model, is_root_package = True):
  this_package_embedding = get_package_embedding(this_package_path, model)
  class_package_paths, subdiv_package_paths = get_subpackages(this_package_path)

  match get_package_type(this_package_path):
    case PackageType.CLASSES_AND_CLASS_PACKAGES:
      add_class_package_on_parent_package_projections(class_package_paths, this_package_embedding, projections, model)
      add_subdiv_package_on_parent_package_projections(subdiv_package_paths, this_package_embedding, projections, model)
    case PackageType.NO_CLASSES_BUT_CLASS_PACKAGES:
      add_class_package_pairwise_projections(class_package_paths, projections, is_root_package, model)
    case PackageType.CLASSES_BUT_NO_CLASS_PACKAGES:
      add_subdiv_package_on_parent_package_projections(subdiv_package_paths, this_package_embedding, projections, model)

  subpackages = next(os.walk(this_package_path))[1]
  for subpackage in subpackages:
    subpackage_path = this_package_path + "/" + subpackage
    find_all_projections_recursively(subpackage_path, projections, model, False)

if __name__ == "__main__":
  # Prompt user to enter path without final slash
  path = input("Enter the path to the Java package of which you want the structure to be analyzed: ")

  # FastText
  import fasttext
  original_directory = os.getcwd()
  os.chdir(os.path.dirname(os.path.abspath(sys.argv[0]))) # Change the working directory to that of the file (so that the relative file path to the model file works consistently)
  model = fasttext.load_model('./models/wiki-news-300d-1M-subword.bin')
  # model = fasttext.load_model('./models/crawl-300d-2M-subword.bin')
  os.chdir(original_directory) # Restore the original working directory so that paths are traversed normally in the rest of the program

  # For each package, derive its embedding from the classes it contains and calculate the projections of the embeddings of its subpackages onto its own embedding
  projections = []
  find_all_projections_recursively(path, projections, model)

  print("Results")
  print("---")
  print("All projection values:", projections)
  if (len(projections) > 0):
    print("Number of projection values:", len(projections))
    print("Minimum:                    ", min(projections))
    print("Maximum:                    ", max(projections))
    print("Average:                    ", np.mean(projections))
    print("Median:                     ", np.median(projections))
    print("Standard deviation:         ", np.std(projections))