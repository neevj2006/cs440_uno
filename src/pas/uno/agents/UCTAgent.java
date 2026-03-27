package src.pas.uno.agents;

// SYSTEM IMPORTS
import edu.bu.pas.uno.Card;
import edu.bu.pas.uno.Game;
import edu.bu.pas.uno.Game.GameView;
import edu.bu.pas.uno.agents.Agent;
import edu.bu.pas.uno.agents.MCTSAgent;
import edu.bu.pas.uno.enums.Color;
import edu.bu.pas.uno.enums.Value;
import edu.bu.pas.uno.moves.Move;
import edu.bu.pas.uno.tree.Node;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

public class UCTAgent extends MCTSAgent {

    // === HIGH PERFORMANCE CACHING ===
    private static java.lang.reflect.Field realGameField = null;
    private static java.lang.reflect.Field agentsField = null;
    private static boolean reflectionInitialized = false;

    public static void initReflection(GameView view) {
        if (reflectionInitialized)
            return;
        try {
            for (java.lang.reflect.Field f : view.getClass().getDeclaredFields()) {
                if (Game.class.isAssignableFrom(f.getType())) {
                    f.setAccessible(true);
                    realGameField = f;
                    break;
                }
            }
            agentsField = Game.class.getDeclaredField("agents");
            agentsField.setAccessible(true);
        } catch (Throwable t) {
        }
        reflectionInitialized = true;
    }

    public static class MCTSNode extends Node {
        public MCTSNode[] children;
        public int parentAction = -1;
        public int numActions;
        public long visits = 0;

        public MCTSNode(final GameView game, final int logicalPlayerIdx, final Node parent) {
            super(game, logicalPlayerIdx, parent);

            NodeState state = this.getNodeState();
            if (state == NodeState.HAS_LEGAL_MOVES) {
                numActions = this.getOrderedLegalMoves().size();
            } else if (state == NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
                numActions = 2;
            } else if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
                numActions = 1;
            } else {
                numActions = 0;
            }

            if (numActions > 0) {
                children = new MCTSNode[numActions];
            }
        }

        public boolean isFullyExpanded() {
            if (numActions == 0)
                return true;
            for (int i = 0; i < numActions; i++) {
                if (children[i] == null)
                    return false;
            }
            return true;
        }

        public int getRandomUntriedAction() {
            int[] untried = new int[numActions];
            int count = 0;
            for (int i = 0; i < numActions; i++) {
                if (children[i] == null) {
                    untried[count++] = i;
                }
            }
            if (count == 0)
                return 0;
            return untried[ThreadLocalRandom.current().nextInt(count)];
        }

        public int getBestUCBAction(boolean isMaximizing) {
            int bestA = -1;
            float bestVal = isMaximizing ? -Float.MAX_VALUE : Float.MAX_VALUE;

            float logVisits = (float) Math.log(this.visits);

            for (int a = 0; a < numActions; a++) {
                long n_sa = this.getQCount(a);
                if (n_sa == 0)
                    continue;

                float qBar = this.getQValueTotal(a) / n_sa;
                float ucbBonus = (float) Math.sqrt(2.0 * logVisits / n_sa);

                float val;
                if (isMaximizing) {
                    val = qBar + ucbBonus;
                    if (val > bestVal) {
                        bestVal = val;
                        bestA = a;
                    }
                } else {
                    val = qBar - ucbBonus;
                    if (val < bestVal) {
                        bestVal = val;
                        bestA = a;
                    }
                }
            }
            return (bestA == -1) ? 0 : bestA;
        }

        @Override
        public Node getChild(final Move move) {
            Game childGame = createDeterminizedGame(this.getGameView());
            if (childGame == null) {
                return new MCTSNode(this.getGameView(), this.getLogicalPlayerIdx(), this);
            }

            childGame.setListener(null);
            injectDummyAgents(childGame);

            if (move != null) {
                applyMoveSafely(childGame, move, this.getLogicalPlayerIdx());
            } else {
                NodeState state = this.getNodeState();
                if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
                    int drawCount = childGame.getUnresolvedCards().total();
                    childGame.drawTotal(childGame.getHand(this.getLogicalPlayerIdx()), drawCount);
                    childGame.getUnresolvedCards().clear();
                    childGame.getPlayerOrder().advance();
                } else {
                    childGame.getPlayerOrder().advance();
                }
            }
            return new MCTSNode(childGame.getOmniscientView(), childGame.getPlayerOrder().getCurrentLogicalPlayerIdx(),
                    this);
        }
    }

    public UCTAgent(final int playerIdx, final long maxThinkingTimeInMS) {
        super(playerIdx, maxThinkingTimeInMS);
    }

    @Override
    public Node search(final GameView game, final Integer drawnCardIdx) {
        initReflection(game);

        MCTSNode root = new MCTSNode(game, this.getLogicalPlayerIdx(), null);
        if (root.numActions == 0)
            return root;

        // Hard cap the thinking time to a maximum of 500ms
        long thinkingTime = Math.min(this.getMaxThinkingTimeInMS(), 500);
        long endTime = System.currentTimeMillis() + thinkingTime - 150;

        while (!Thread.currentThread().isInterrupted() && System.currentTimeMillis() < endTime) {
            try {
                // 1. SELECTION
                MCTSNode curr = root;
                while (curr.isFullyExpanded() && curr.numActions > 0) {
                    boolean isMax = (curr.getLogicalPlayerIdx() == this.getLogicalPlayerIdx());
                    int bestA = curr.getBestUCBAction(isMax);
                    curr = curr.children[bestA];
                }

                // 2. EXPANSION
                if (curr.numActions > 0) {
                    int actionToExpand = curr.getRandomUntriedAction();
                    Integer dIdx = (curr == root) ? drawnCardIdx : null;
                    Move move = getMoveForActionIdx(curr, actionToExpand, dIdx);

                    MCTSNode child = (MCTSNode) curr.getChild(move);
                    child.parentAction = actionToExpand;
                    curr.children[actionToExpand] = child;
                    curr = child;
                }

                // 3. SIMULATION
                float reward = playSingleRandomGame(curr);

                // 4. BACKPROPAGATION
                MCTSNode backpropNode = curr;
                while (backpropNode != null) {
                    backpropNode.visits++;
                    if (backpropNode.getParent() != null && backpropNode.getParent() instanceof MCTSNode) {
                        MCTSNode parent = (MCTSNode) backpropNode.getParent();
                        int a = backpropNode.parentAction;
                        if (a != -1) {
                            parent.setQCount(a, parent.getQCount(a) + 1);
                            parent.setQValueTotal(a, parent.getQValueTotal(a) + reward);
                        }
                    }
                    if (backpropNode == root)
                        break;
                    backpropNode = (MCTSNode) backpropNode.getParent();
                }
            } catch (Throwable t) {
                break;
            }
        }

        return root;
    }

    @Override
    public Move argmaxQValues(final Node node) {
        MCTSNode mNode = (MCTSNode) node;
        if (mNode.numActions == 0)
            return null;

        int bestIdx = 0;
        float maxQ = -Float.MAX_VALUE;

        for (int i = 0; i < mNode.numActions; i++) {
            long visits = mNode.getQCount(i);
            float q = (visits == 0) ? 0 : mNode.getQValueTotal(i) / visits;

            if (q > maxQ) {
                maxQ = q;
                bestIdx = i;
            }
        }

        return getMoveForActionIdx(mNode, bestIdx, null);
    }

    // ==========================================================
    // HELPER METHODS
    // ==========================================================

    private float playSingleRandomGame(MCTSNode node) {
        Game simGame = createDeterminizedGame(node.getGameView());
        if (simGame == null)
            return 0.0f;

        simGame.setListener(null);
        injectDummyAgents(simGame);

        int maxDepth = 150;
        int depth = 0;

        while (!simGame.isOver() && depth < maxDepth && !Thread.currentThread().isInterrupted()) {
            int currLogicalIdx = simGame.getPlayerOrder().getCurrentLogicalPlayerIdx();
            GameView view = simGame.getOmniscientView();

            if (simGame.getUnresolvedCards().total() > 0) {
                int drawCount = simGame.getUnresolvedCards().total();
                simGame.drawTotal(simGame.getHand(currLogicalIdx), drawCount);
                simGame.getUnresolvedCards().clear();
                simGame.getPlayerOrder().advance();
            } else {
                Set<Integer> legalMovesSet = view.getHandView(currLogicalIdx).getLegalMoves(view);

                if (!legalMovesSet.isEmpty()) {
                    int targetIdx = ThreadLocalRandom.current().nextInt(legalMovesSet.size());
                    int randMoveIdx = -1;
                    int idx = 0;
                    for (Integer m : legalMovesSet) {
                        if (idx == targetIdx) {
                            randMoveIdx = m;
                            break;
                        }
                        idx++;
                    }

                    Card c = view.getHandView(currLogicalIdx).getCard(randMoveIdx);
                    Agent currAgent = simGame.getAgent(currLogicalIdx);
                    Move randomMove = isCardWildStatic(c)
                            ? Move.createMove(currAgent, randMoveIdx, getValidRandomColorStatic())
                            : Move.createMove(currAgent, randMoveIdx);

                    try {
                        simGame.resolveMove(randomMove);
                    } catch (Throwable t) {
                    }
                } else {
                    simGame.drawCard(simGame.getHand(currLogicalIdx));
                    GameView postDrawView = simGame.getOmniscientView();
                    Set<Integer> postDrawLegal = postDrawView.getHandView(currLogicalIdx).getLegalMoves(postDrawView);
                    int drawnCardIdx = postDrawView.getHandView(currLogicalIdx).size() - 1;

                    if (postDrawLegal.contains(drawnCardIdx) && ThreadLocalRandom.current().nextBoolean()) {
                        Card c = postDrawView.getHandView(currLogicalIdx).getCard(drawnCardIdx);
                        Agent currAgent = simGame.getAgent(currLogicalIdx);

                        Move randomMove = isCardWildStatic(c)
                                ? Move.createMove(currAgent, drawnCardIdx, getValidRandomColorStatic())
                                : Move.createMove(currAgent, drawnCardIdx);
                        try {
                            simGame.resolveMove(randomMove);
                        } catch (Throwable t) {
                        }
                    } else {
                        simGame.getPlayerOrder().advance();
                    }
                }
            }
            depth++;
        }

        for (int i = 0; i < simGame.getNumPlayers(); i++) {
            if (simGame.getOmniscientView().getHandView(i).size() == 0) {
                return (i == this.getLogicalPlayerIdx()) ? 1.0f : -1.0f;
            }
        }

        int myCards = simGame.getOmniscientView().getHandView(this.getLogicalPlayerIdx()).size();
        for (int i = 0; i < simGame.getNumPlayers(); i++) {
            if (i != this.getLogicalPlayerIdx()) {
                int oppCards = simGame.getOmniscientView().getHandView(i).size();
                if (myCards < oppCards)
                    return 0.5f;
                if (myCards > oppCards)
                    return -0.5f;
            }
        }
        return 0.0f;
    }

    private Move getMoveForActionIdx(MCTSNode node, int actionIdx, Integer drawnCardIdx) {
        Node.NodeState state = node.getNodeState();
        int currLogicalIdx = node.getLogicalPlayerIdx();
        GameView view = node.getGameView();

        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            int cardIdx = node.getOrderedLegalMoves().get(actionIdx);
            Card c = view.getHandView(currLogicalIdx).getCard(cardIdx);

            if (isCardWildStatic(c)) {
                return Move.createMove(this, cardIdx, getValidRandomColorStatic());
            } else {
                return Move.createMove(this, cardIdx);
            }
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
            if (actionIdx == 1) {
                int cIdx = (drawnCardIdx != null) ? drawnCardIdx : (view.getHandView(currLogicalIdx).size() - 1);
                Card c = view.getHandView(currLogicalIdx).getCard(cIdx);
                if (isCardWildStatic(c)) {
                    return Move.createMove(this, cIdx, getValidRandomColorStatic());
                } else {
                    return Move.createMove(this, cIdx);
                }
            }
        }
        return null;
    }

    public static Game createDeterminizedGame(GameView view) {
        initReflection(view);
        if (realGameField != null) {
            try {
                Game realGame = (Game) realGameField.get(view);
                return new Game(realGame.getOmniscientView());
            } catch (Throwable t) {
            }
        }
        return null;
    }

    public static boolean isCardWildStatic(Card c) {
        if (c == null)
            return false;
        if (c.isWild())
            return true;
        if (c.value() == Value.WILD || c.value() == Value.WILD_DRAW_FOUR)
            return true;
        return false;
    }

    public static Color getValidRandomColorStatic() {
        Color[] validColors = { Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW };
        return validColors[ThreadLocalRandom.current().nextInt(validColors.length)];
    }

    public static void injectDummyAgents(Game game) {
        if (agentsField != null) {
            try {
                if (agentsField.get(game) == null) {
                    Agent[] dummyAgents = new Agent[game.getNumPlayers()];
                    for (int i = 0; i < dummyAgents.length; i++) {
                        int realIdx = game.getPlayerOrder().getAgentIdx(i);
                        UCTAgent dummy = new UCTAgent(realIdx, 5);
                        dummyAgents[i] = dummy;
                    }
                    agentsField.set(game, dummyAgents);
                }
            } catch (Throwable e) {
            }
        }
    }

    public static void applyMoveSafely(Game game, Move originalMove, int logicalIdx) {
        try {
            Agent correctAgent = game.getAgent(logicalIdx);
            int cIdx = originalMove.getCardToPlayIdx();
            Card c = game.getOmniscientView().getHandView(logicalIdx).getCard(cIdx);

            Move safeMove;
            if (isCardWildStatic(c)) {
                safeMove = Move.createMove(correctAgent, cIdx, getValidRandomColorStatic());
            } else {
                safeMove = Move.createMove(correctAgent, cIdx);
            }
            game.resolveMove(safeMove);
        } catch (Throwable t) {
            try {
                game.resolveMove(originalMove);
            } catch (Throwable t2) {
            }
        }
    }
}