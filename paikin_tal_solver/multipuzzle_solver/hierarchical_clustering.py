import copy
import logging

import sys

from hammoudeh_puzzle import config


class SegmentCluster(object):

    def __init__(self, cluster_id, pieces):
        """
        Segment Cluster Constructor.

        Args:
            cluster_id (int): Identification number of the clusters.
            pieces (List[int]): List of pieces in the cluster.
        """
        self._segment_ids = [cluster_id]
        self._cluster_id = cluster_id
        self._pieces = copy.copy(pieces)
        self._merging_allowed = True

        # History of the previous clusters that were merged.
        self._merging_history = []

    @property
    def id_number(self):
        """
        Identification number of the cluster.

        Returns (int): Cluster identification number.
        """
        return self._cluster_id

    def get_pieces(self):
        """
        Gets the pieces in the segment cluster.

        Returns (List[int]): List of the pieces (by identification number) in the cluster.
        """
        return copy.copy(self._pieces)

    @property
    def is_merging_allowed(self):
        """
        Gets whether this cluster can be merged with other clusters.

        Returns (bool): True if this cluster can be merged with others and False otherwise.
        """
        return self._merging_allowed

    def get_segments(self):
        """
        Gets a list of the one or more segments that comprise the cluster.

        Returns (List[int]): Solver segment(s) in the cluster.
        """
        return copy.copy(self._segment_ids)

    def disallow_merging(self):
        """
        Disallows merging of this cluster with any other clusters.
        """
        self._merging_allowed = False

    @staticmethod
    def merge_clusters(first_cluster, second_cluster, intercluster_similarity):
        """
        Factory method that takes two clusters to be merged and returns

        Args:
            first_cluster (SegmentCluster): First cluster to be merged
            second_cluster (SegmentCluster): Other cluster to be merged
            intercluster_similarity (float): Cluster similarity between the first and second cluster.

        Returns (SegmentCluster):
            A merged cluster containing the the first and segment cluster merged together.
        """

        # Build the merged cluster
        new_cluster_id = min(first_cluster.id_number, second_cluster.id_number)
        merged_piece_list = first_cluster._pieces + second_cluster._pieces
        merged_cluster = SegmentCluster(new_cluster_id, merged_piece_list)

        # Define the segments in this cluster.
        merged_cluster._segment_ids = sorted(first_cluster._segment_ids + second_cluster._segment_ids)

        # Make the history object for this merged.
        merging_history = [(merged_cluster._segment_ids, intercluster_similarity)]

        # Combine the merging history
        merging_history += first_cluster._merging_history + second_cluster._merging_history
        merged_cluster._merging_history = sorted(merging_history, key=lambda history: history[1])

        return merged_cluster


class HierarchicalClustering(object):

    MINIMUM_CLUSTER_SIMILARITY = 0.1

    @staticmethod
    def run(segments, similarity_matrix):
        """
        Performs the hierarchical cluster.

        Args:
            segments (List[PuzzleSegment]): List of segments in the puzzle.
            similarity_matrix (Numpy[float]): A similarity matrix showing the similarity between the segments.

        Returns (List[SegmentCluster]]): Merged clusters in the puzzle.
        """

        # Build the clusters
        clusters = [SegmentCluster(segment.id_number, segment.get_piece_ids()) for segment in segments]

        while True:
            matrix_shape = similarity_matrix.shape

            # Verify the shape makes sense.
            if config.PERFORM_ASSERT_CHECKS:
                assert matrix_shape[0] == matrix_shape[1]

            max_dist = -sys.maxint
            clusters_to_merge = None
            for row in xrange(0, len(clusters)):

                # Check if this cluster is prevented from further merging.
                if not clusters[row].is_merging_allowed:
                    continue

                for col in xrange(row + 1, len(clusters)):

                    # Check if this cluster is prevented from further merging.
                    if not clusters[col].is_merging_allowed:
                        continue

                    # Determine if the two clusters should be merged.
                    if similarity_matrix[row, col] > max_dist \
                            and HierarchicalClustering._allow_cluster_merging(clusters, row, col, similarity_matrix):
                        max_dist = HierarchicalClustering._get_intercluster_similarity(similarity_matrix,
                                                                                       row, col)
                        clusters_to_merge = (row, col)

            # If no clusters to merge, return the clusters
            if clusters_to_merge is None:
                return clusters, similarity_matrix

            # Update clusters and similarity matrix
            clusters, similarity_matrix = HierarchicalClustering.merge_clusters(clusters,
                                                                                clusters_to_merge[0],
                                                                                clusters_to_merge[1],
                                                                                similarity_matrix)

    @staticmethod
    def merge_clusters(clusters, first_cluster_id, second_cluster_id, similarity_matrix):
        """
        Merges the two specified clusters and updates the cluster list as well as the similarity matrix.

        Args:
            clusters (List[SegmentCluster]): All current clusters.
            first_cluster_id (int): Identification of the first cluster to merge.
            second_cluster_id (int): Identification of the second cluster to merge.
            similarity_matrix (Numpy[float]): Matrix containing the similarity between all clusters.

        Returns (Tuple[List[SegmentCluster], Numpy[float]):
            A list of the new clusters and the updated similarity matrix.
        """

        # Get the minimum and maximum index
        [min_index, second_index] = sorted([first_cluster_id, second_cluster_id])

        logging.info("Merging clusters #%d and #%d together." % (min_index, second_index))

        # Build the merged cluster.
        clusters[min_index] = SegmentCluster.merge_clusters(clusters[first_cluster_id], clusters[second_cluster_id],
                                                            similarity_matrix[min_index, second_index])
        # Replace the other cluster that was merged.
        last_cluster_id = len(clusters) - 1
        clusters[second_index] = clusters[last_cluster_id]

        # Update the similarity matrix
        for other_cluster_id in xrange(0, len(clusters)):
            # Skip self comparison
            if other_cluster_id == min_index or other_cluster_id == second_index:
                continue
            # Get the maximum similarity
            first_cluster_similarity = HierarchicalClustering._get_intercluster_similarity(similarity_matrix,
                                                                                           min_index,
                                                                                           other_cluster_id)
            second_cluster_similarity = HierarchicalClustering._get_intercluster_similarity(similarity_matrix,
                                                                                            second_index,
                                                                                            other_cluster_id)
            intercluster_similarity = max(first_cluster_similarity, second_cluster_similarity)

            # Update the similarity
            HierarchicalClustering._update_intercluster_similarity(similarity_matrix, first_cluster_id,
                                                                   other_cluster_id, intercluster_similarity)

        # Merge in the last cluster
        for other_cluster_id in xrange(0, last_cluster_id):
            # Skip the cluster being merged into
            if other_cluster_id == second_index:
                continue
            # Get and update the similarity for the last cluster.
            intercluster_similarity = HierarchicalClustering._get_intercluster_similarity(similarity_matrix,
                                                                                          last_cluster_id,
                                                                                          other_cluster_id)
            # Use second index as the destination since folding the matrix
            HierarchicalClustering._update_intercluster_similarity(similarity_matrix, second_index,
                                                                   other_cluster_id, intercluster_similarity)

        # Remove the last cluster
        return clusters[:last_cluster_id], similarity_matrix[:last_cluster_id, :last_cluster_id]

    @staticmethod
    def _get_intercluster_similarity(similarity_matrix, first_cluster_id, second_cluster_id):
        """
        Gets the similarity between clusters.

        Args:
            similarity_matrix (Numpy[float): Similarity between all clusters
            first_cluster_id (int): Identification number of the first cluster
            second_cluster_id (int): Identification number of the second cluster
        """
        if first_cluster_id == second_cluster_id:
            raise ValueError("The cluster identification numbers cannot be identical.")

        [row, col] = sorted([first_cluster_id, second_cluster_id])
        return similarity_matrix[row, col]

    @staticmethod
    def _update_intercluster_similarity(similarity_matrix, first_cluster_id, second_cluster_id, value):
        """
        Updates the similarity value between the two specified clusters.

        Args:
            similarity_matrix (Numpy[float): Similarity between all clusters
            first_cluster_id (int): Identification number of the first cluster
            second_cluster_id (int): Identification number of the second cluster
            value (float): New intercluster similarity.
        """
        if first_cluster_id == second_cluster_id:
            raise ValueError("The cluster identification numbers cannot be identical.")

        [row, col] = sorted([first_cluster_id, second_cluster_id])
        similarity_matrix[row, col] = value

    # noinspection PyUnusedLocal
    @staticmethod
    def _allow_cluster_merging(clusters, first_cluster_id, second_cluster_id, similarity_matrix):
        """
        Allows for more sophisticated checking of the clustering.

        Args:
            clusters (List[SegmentCluster]): Segment clusters.
            first_cluster_id (int): Identification number of the first cluster to merge
            second_cluster_id (int): Identification number of the second cluster to merge.
            similarity_matrix (Numpy[float]): Similarity matrix between clusters.

        Returns (bool):
            True if the clusters can be merged and False otherwise.
        """
        intercluster_similarity = HierarchicalClustering._get_intercluster_similarity(similarity_matrix,
                                                                                      first_cluster_id,
                                                                                      second_cluster_id)

        # Cluster must exceed some minimum similarity.
        if intercluster_similarity < HierarchicalClustering.MINIMUM_CLUSTER_SIMILARITY:
            return False
        return True
