import sys
import os
import numpy as np
from anytree import Node, RenderTree
from common import get_package_embedding, calculate_projection_length

DEBUG = False

def print_debug(*args, num_newlines=2, sep=" ", end="\n"):
  if DEBUG:
    print("\n" * num_newlines, end="")
    print(*args, sep=sep, end=end)

if __name__ == "__main__":
  # Prompt user to enter path without final slash
  path = input("Enter the path to the Java package that needs to be organized: ")
  projection_threshold = float(input("Enter projection threshold at which a package should be considered a subpackage: "))

  packages = next(os.walk(path))[1]

  # FastText
  import fasttext
  original_directory = os.getcwd()
  os.chdir(os.path.dirname(os.path.abspath(sys.argv[0]))) # Change the working directory to that of the file (so that the relative file path to the model file works consistently)
  # model = fasttext.load_model('./models/crawl-300d-2M-subword.bin')
  model = fasttext.load_model('./models/wiki-news-300d-1M-subword.bin')
  os.chdir(original_directory) # Restore the original working directory so that paths are traversed normally in the rest of the program
  
  # For each package, derive its embedding from the classes it contains
  package_embeddings = {}
  for i in range(len(packages)):
    package = packages[i]
    package_path = path + "/" + package
    package_embedding = get_package_embedding(package_path, model)
    if package_embedding is not None:
      package_embeddings[package_path] = package_embedding

  # Find embedding for root package
  root_package_embedding = get_package_embedding(path, model)
  
  ##########################
  # Architecture discovery #
  ##########################
  """
  Dictionary of group-package pairs `((pkg1, ...), pkg2)` to the length of the projection of `(pkg1, ...)` onto `pkg2`
  """
  pairwise_projections = {}

  """
  Used to keep track of the package hierarchy during the algorithm.
  
  Example:
    is_contained_in[(pkg1, ...)] == pkg2

    (pkg1, ...) is a subpackage of pkg2
  """
  is_contained_in = {} 

  """
  If multiple projections exist for the same subpackage, store the less-valued ones for later

  Example:
    Projection 1: 'bread' projected onto 'sandwich': 0.95
    Projection 2: 'bread' projected onto 'lunch': 0.9

    Initially 'bread' might be set such that it is a subpackage of 'sandwich'. In that case, projection 2
     is obviously put aside, because 'bread' is already put inside a parent package, even if projection 2
     is the next highest valued projection. However, if it so happens that 'bread' is part of a loop
     structure, then all the packages that are part of this structure should be considered as one group. 
     However, this group has no parent package at the time of discovery. Let's say that 'bread' and 
     'sandwich' form a loop structure. In that case, we would like to reinstate projection 2.
  """
  backup_projections = {}
  
  """
  Used to keep track of which packages are grouped together
  
  Example:
    group_map[(pkg1,)] == (pkg1,pkg2,pkg3)

    pkg1 was grouped together with pkg2 and pkg3 because they formed a loop hierarchy.

  Example 2:
    group_map[(pkg1,)] == (pkg1,)

    pkg1 was not grouped together with other packages and simply forms its own group on its own
  """
  group_map = {}

  for this_package, this_package_embedding in package_embeddings.items():
    # For each package, initialize the variables
    this_package_as_tuple = (this_package,)
    is_contained_in[this_package_as_tuple] = None
    backup_projections[this_package_as_tuple] = []
    group_map[this_package_as_tuple] = this_package_as_tuple

    # If the root package has an embedding, add projections of subpackages onto root package
    if root_package_embedding is not None: 
      pairwise_projections[(this_package_as_tuple, path)] = calculate_projection_length(this_package_embedding, root_package_embedding)

    ## Calculate pairwise projections 
    for that_package, that_package_embedding in package_embeddings.items():
      if this_package == that_package:
        continue
      pair = (this_package_as_tuple, that_package)
      pairwise_projections[pair] = calculate_projection_length(this_package_embedding, that_package_embedding)

  # If the root package has an embedding, we need to initialize the variables corresponding to the root package as well
  if root_package_embedding is not None:
    group_map[(path,)] = (path,)
    is_contained_in[(path,)] = None

  pairwise_projections = dict(sorted(pairwise_projections.items(), key=lambda item: item[1], reverse=True))
  
  # Recommend a hierarchy using the PackageEmbedding-on-PackageEmbedding projections
  queue = pairwise_projections.copy()
  while (len(queue) > 0):
    projection = next(iter(queue))
    projection_child, projection_parent = projection
    value = queue.pop(projection)

    if (value >= projection_threshold):
      if is_contained_in[projection_child] is None:
        is_contained_in[projection_child] = projection_parent
        
        # Test for loop
        parent = (projection_parent,)
        seen = [parent]

        print_debug("Projection_child is", projection_child, "and projection_parent is", projection_parent)
        
        while is_contained_in[group_map[parent]] is not None:
          parent = (is_contained_in[group_map[parent]],)
          if (parent[0] in projection_child):
            print_debug("Detected a loop. More specifically", parent[0], "was found in", projection_child)
            print_debug("Seen is", seen)
            print_debug("group_map is", group_map)

            # There is a loop in the current package hierarchy
            # Each package in the loop structure should not be contained in another package on its own
            involved = set(map(lambda x : group_map[x], seen))

            print_debug("The involved packages before adding projection_child are:", involved)

            involved.add(projection_child)

            print_debug("The involved packages are:", involved)
            print_debug("is_contained_in before removing the old packages was:", is_contained_in)
              
            for g in involved:
              is_contained_in.pop(g)

            print_debug("is_contained_in after removing the old packages is:", is_contained_in)

            # Instead all of the packages in the loop structure should form a group and the entire group is contained in some other package
            group_tuple = tuple({item for tup in involved for item in tup}) # flatten
            is_contained_in[group_tuple] = None
            
            print_debug("is_contained_in after adding the group tuple is:", is_contained_in)

            # Update the queue to reflect the creation of the group
            _queue = queue.copy()
            _replaced_keys = set()
            _parent_to_projections_map = {}
            for _projection_with_value in _queue.items():
              _projection, _ = _projection_with_value
              _projection_child, _projection_parent = _projection
              if _projection_child in involved:
                _replaced_keys.add(_projection_child)
              
                # For each `(child, parent): value` item where `child` was part of the loop hierarchy, remove that item and add it to `_parent_to_projections_map` for it to be averaged later
                queue.pop(_projection)

                print_debug("found an item in the queue:", _projection_with_value)

                if _projection_parent not in group_tuple: # a group cannot be a subpackage of one of its members
                  print_debug("the parent is not in group_tuple, so we add it to _parent_to_projections_map")
                  _parent_to_projections_map.setdefault(_projection_parent, []).append(_projection_with_value)
                  print_debug("_parent_to_projections_map now looks like:", _parent_to_projections_map)
            
            # Also, take into account the backup projections that involve one of the affected packages / package groups
            print_debug("backup projections before adding back elements was", backup_projections)

            for pkg in involved:
              for backup_projection_with_value in backup_projections[pkg]:
                backup_projection, _ = backup_projection_with_value
                backup_projection_parent = backup_projection[1]

                print_debug("found an item in backup list:", backup_projection_with_value)
                
                if backup_projection_parent not in group_tuple: # a group cannot be a subpackage of one of its members
                  print_debug("the parent is not in group_tuple, so we add it to _parent_to_projections_map")
                  _parent_to_projections_map.setdefault(backup_projection_parent, []).append(backup_projection_with_value)
                  print_debug("_parent_to_projections_map now looks like:", _parent_to_projections_map)
                  
            # Calculate all average projection values and add back to queue as (group_tuple, parent): average_projection_value
            for parent, list_of_projections in _parent_to_projections_map.items():
              average_projection_value = np.average(list(map(lambda x : x[1], list_of_projections)))
              queue[(group_tuple, parent)] = average_projection_value

            # And each individual package `pkg` should now be mapped to the group they comprise, because we need to look at is_contained_in(group_tuple) instead of is_contained_in(pkg)
            for pkg in group_tuple:
              group_map[(pkg,)] = group_tuple

            # For all restored projections that are now both in backup_projections and queue, remove them from backup_projections
            for _projection_child in _replaced_keys:
              backup_projections.pop(_projection_child)
            # Also, facilitate backup projections for the new group
            backup_projections[group_tuple] = []
            
            print_debug("backup projections after adding back elements was", backup_projections)
            print_debug("and now queue is", queue)
            
            # if new items were added, the dict needs to be resorted, because the new items have been added to the end 
            queue = dict(sorted(queue.items(), key=lambda item: item[1], reverse=True))
            
            print_debug("and after sorting, queue is", queue)

            break
          else:
            seen.append(parent)
      else:
        # if the child already has a parent, save this item for later in case the child is part of a loop hierarchy
        backup_projections[projection_child].append((projection, value))

####################################
# Display result in tree structure #
####################################
def add_package_to_tree(child, parent, package_to_node_map):
  if child not in package_to_node_map:
    package_to_node_map[child] = Node(child, parent=parent)
  else:
    package_to_node_map[child].parent = parent

# Construct the tree
root = Node(path)
package_to_node_map = {}
package_to_node_map[None] = root
package_to_node_map[path] = root
for children, parent in is_contained_in.items():
  if parent not in package_to_node_map:
    package_to_node_map[parent] = Node(parent)
  if len(children) > 1: # subpackage containing multiple packages
    subpackage = Node("•", parent=package_to_node_map[parent])
    for child in children:
      add_package_to_tree(child, subpackage, package_to_node_map)
  else: # singular package
    if children[0] != path: # the root package is not contained in anything (None). Since None maps to the root package, we have to make sure that the root package is not added as a child node of itself.
      add_package_to_tree(children[0], package_to_node_map[parent], package_to_node_map)

# Mark lax subpackages
if root_package_embedding is not None:
  for subpackage_node in root.children:
    projection_length = np.mean([pairwise_projections[((subpackage_child_node.name,), path)] for subpackage_child_node in subpackage_node.children]) if subpackage_node.name == "•" else pairwise_projections[((subpackage_node.name,), path)]
    if projection_length < projection_threshold:
      subpackage_node.name += " (lax, projection value = " + str(projection_length) + ")" 

# Print result
print("Recommended package structure of \"" + os.path.basename(path) + "\" with threshold = " + str(projection_threshold) + ":")
for pre, fill, node in RenderTree(root):
  print("%s%s" % (pre, os.path.basename(node.name)))