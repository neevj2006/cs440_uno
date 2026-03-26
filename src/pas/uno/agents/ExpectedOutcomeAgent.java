package src.pas.uno.agents;

// SYSTEM IMPORTS
import edu.bu.pas.uno.Card;
import edu.bu.pas.uno.Game;
import edu.bu.pas.uno.Game.GameView;
import edu.bu.pas.uno.Hand.HandView;
import edu.bu.pas.uno.agents.Agent;
import edu.bu.pas.uno.agents.MCTSAgent;
import edu.bu.pas.uno.enums.Color;
import edu.bu.pas.uno.enums.Value;
import edu.bu.pas.uno.moves.Move;
import edu.bu.pas.uno.tree.Node;

import java.util.List;
import java.util.Random;

// JAVA PROJECT IMPORTS

public class ExpectedOutcomeAgent extends MCTSAgent {

    public static class MCTSNode extends Node {
        public MCTSNode(final GameView game, final int logicalPlayerIdx, final Node parent) {
            super(game, logicalPlayerIdx, parent);
        }

        @Override
        public Node getChild(final Move move) {
            Game childGame = new Game(this.getGameView());
            childGame.setListener(null);

            try {
                if (move != null) {
                    childGame.resolveMove(move);
                } else {
                    NodeState state = this.getNodeState();
                    if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
                        int drawCount = childGame.getUnresolvedCards().total();
                        childGame.drawTotal(childGame.getHand(this.getLogicalPlayerIdx()), drawCount);
                        childGame.getUnresolvedCards().clear(); // CRITICAL: Clear the buffer!
                        childGame.getPlayerOrder().advance();
                    } else if (state == NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
                        childGame.getPlayerOrder().advance();
                    } else {
                        childGame.getPlayerOrder().advance();
                    }
                }
            } catch (Exception e) {
                System.err.println("[DEBUG-ERROR] Exception in getChild: " + e.getMessage());
                e.printStackTrace();
            }

            return new MCTSNode(childGame.getOmniscientView(), childGame.getPlayerOrder().getCurrentLogicalPlayerIdx(),
                    this);
        }
    }

    public ExpectedOutcomeAgent(final int playerIdx, final long maxThinkingTimeInMS) {
        super(playerIdx, maxThinkingTimeInMS);
    }

    @Override
    public Node search(final GameView game, final Integer drawnCardIdx) {
        System.out.println("\n[DEBUG-search] === Starting MCTS Search ===");
        MCTSNode root = new MCTSNode(game, this.getLogicalPlayerIdx(), null);

        int numActions = 0;
        Node.NodeState state = root.getNodeState();

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            numActions = root.getOrderedLegalMoves().size();
            System.out.println("[DEBUG-search] Root state: HAS_LEGAL_MOVES. Total actions: " + numActions);
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            numActions = 2;
            System.out.println("[DEBUG-search] Root state: MAY_PLAY_DRAWN_CARD. Total actions: 2");
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            numActions = 1;
            System.out.println("[DEBUG-search] Root state: UNRESOLVED_CARDS_PRESENT. Total actions: 1");
        }

        if (numActions == 0)
            return root;

        // BUMPED BUFFER: 150ms buffer to guarantee we return before the framework kills
        // the thread
        long endTime = System.currentTimeMillis() + this.getMaxThinkingTimeInMS() - 150;

        Node[] children = new Node[numActions];
        try {
            for (int i = 0; i < numActions; i++) {
                Move move = getMoveForActionIdx(root, i, drawnCardIdx);
                children[i] = root.getChild(move);
            }
        } catch (Exception e) {
            System.err.println("[DEBUG-ERROR] Exception while pre-computing children!");
            e.printStackTrace();
            return root; // Return early if it fails
        }

        int totalSims = 0;

        System.out.println("[DEBUG-search] Running round-robin rollout simulations...");
        while (!Thread.currentThread().isInterrupted() && System.currentTimeMillis() < endTime) {
            for (int actionIdx = 0; actionIdx < numActions; actionIdx++) {
                if (System.currentTimeMillis() >= endTime || Thread.currentThread().isInterrupted())
                    break;

                try {
                    Game simGame = new Game(children[actionIdx].getGameView());
                    simGame.setListener(null);

                    float utility = playRandomGame(simGame);

                    root.setQValueTotal(actionIdx, root.getQValueTotal(actionIdx) + utility);
                    root.setQCount(actionIdx, root.getQCount(actionIdx) + 1);
                    totalSims++;
                } catch (Exception e) {
                    System.err.println("[DEBUG-ERROR] Exception during random rollout on action " + actionIdx);
                    e.printStackTrace();
                    // Break out of loop to ensure we return safely
                    break;
                }
            }
        }

        System.out.println("[DEBUG-search] Completed " + totalSims + " total simulations.");
        for (int i = 0; i < numActions; i++) {
            float avgQ = root.getQCount(i) == 0 ? 0 : root.getQValueTotal(i) / root.getQCount(i);
            System.out.println(
                    "[DEBUG-search] Action " + i + " -> Count: " + root.getQCount(i) + ", Avg Q-Value: " + avgQ);
        }
        return root;
    }

    @Override
    public Move argmaxQValues(final Node node) {
        System.out.println("[DEBUG-argmax] Evaluating best move...");
        Node.NodeState state = node.getNodeState();
        int numActions = 0;

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            numActions = node.getOrderedLegalMoves().size();
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            numActions = 2;
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
            numActions = 1;
        }

        if (numActions == 0)
            return null;

        int bestIdx = 0;
        float maxQ = -Float.MAX_VALUE;

        for (int i = 0; i < numActions; i++) {
            float q = node.getQValue(i);
            if (q > maxQ) {
                maxQ = q;
                bestIdx = i;
            }
        }

        Move finalMove = getMoveForActionIdx(node, bestIdx, null);
        System.out.println("[DEBUG-argmax] Chose action index " + bestIdx + " -> Final Move: " + finalMove);
        return finalMove;
    }

    // --- HELPER METHODS ---

    private Move getMoveForActionIdx(Node node, int actionIdx, Integer drawnCardIdx) {
        Node.NodeState state = node.getNodeState();
        int currLogicalIdx = node.getLogicalPlayerIdx();
        GameView view = node.getGameView();

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            int cardIdx = node.getOrderedLegalMoves().get(actionIdx);
            Card c = view.getHandView(currLogicalIdx).getCard(cardIdx);

            if (c.isWild()) {
                return Move.createMove(this, cardIdx, Color.getRandomColor(this.getRandom()));
            } else {
                return Move.createMove(this, cardIdx);
            }
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            if (actionIdx == 1) {
                int cIdx = (drawnCardIdx != null) ? drawnCardIdx : (view.getHandView(currLogicalIdx).size() - 1);
                Card c = view.getHandView(currLogicalIdx).getCard(cIdx);
                if (c.isWild()) {
                    return Move.createMove(this, cIdx, Color.getRandomColor(this.getRandom()));
                } else {
                    return Move.createMove(this, cIdx);
                }
            } else {
                return null;
            }
        }

        return null;
    }

    private float playRandomGame(Game simGame) {
        int maxDepth = 250;
        int depth = 0;

        while (!simGame.isOver() && depth < maxDepth && !Thread.currentThread().isInterrupted()) {
            int currLogicalIdx = simGame.getPlayerOrder().getCurrentLogicalPlayerIdx();
            GameView view = simGame.getOmniscientView();
            Node simNode = new MCTSNode(view, currLogicalIdx, null);
            Node.NodeState state = simNode.getNodeState();

            Move randomMove = null;

            if (state == Node.NodeState.HAS_LEGAL_MOVES) {
                List<Integer> legal = simNode.getOrderedLegalMoves();
                int randMoveIdx = legal.get(this.getRandom().nextInt(legal.size()));
                Card c = view.getHandView(currLogicalIdx).getCard(randMoveIdx);

                // Get the proper Agent for this turn, fallback to 'this' if null
                Agent currentAgent = simGame.getAgent(currLogicalIdx);
                if (currentAgent == null)
                    currentAgent = this;

                if (c.isWild()) {
                    randomMove = Move.createMove(currentAgent, randMoveIdx, Color.getRandomColor(this.getRandom()));
                } else {
                    randomMove = Move.createMove(currentAgent, randMoveIdx);
                }
                simGame.resolveMove(randomMove);

            } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
                boolean playDrawnCard = this.getRandom().nextBoolean();
                if (playDrawnCard) {
                    int lastIdx = view.getHandView(currLogicalIdx).size() - 1;
                    Card c = view.getHandView(currLogicalIdx).getCard(lastIdx);

                    Agent currentAgent = simGame.getAgent(currLogicalIdx);
                    if (currentAgent == null)
                        currentAgent = this;

                    if (c.isWild()) {
                        randomMove = Move.createMove(currentAgent, lastIdx, Color.getRandomColor(this.getRandom()));
                    } else {
                        randomMove = Move.createMove(currentAgent, lastIdx);
                    }
                    simGame.resolveMove(randomMove);
                } else {
                    simGame.getPlayerOrder().advance();
                }

            } else if (state == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
                int drawCount = simGame.getUnresolvedCards().total();
                simGame.drawTotal(simGame.getHand(currLogicalIdx), drawCount);
                simGame.getUnresolvedCards().clear(); // CRITICAL: Clear the buffer
                simGame.getPlayerOrder().advance();
            }
            depth++;
        }

        for (int i = 0; i < simGame.getNumPlayers(); i++) {
            if (simGame.getOmniscientView().getHandView(i).size() == 0) {
                if (i == this.getLogicalPlayerIdx()) {
                    return 1.0f; // Win
                } else {
                    return -1.0f; // Loss
                }
            }
        }

        return 0.0f; // Tie
    }
}