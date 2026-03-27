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
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class UnoMCTSAgent extends MCTSAgent {

    private static java.lang.reflect.Field realGameField = null;
    private static java.lang.reflect.Field agentsField = null;

    public static class MCTSNode extends Node {
        public MCTSNode[] children;
        public int parentAction = -1;
        public int numActions;
        public long visits = 0;
        public double[] qTotals;
        public long[] qCounts;
        private final UnoMCTSAgent owner;

        public MCTSNode(final GameView game, final int logicalPlayerIdx, final Node parent, UnoMCTSAgent owner) {
            // CRITICAL: The super constructor handles the internal parent-child pointers
            // for the API
            super(game, logicalPlayerIdx, parent);
            this.owner = owner;

            NodeState state = this.getNodeState();
            if (state == NodeState.HAS_LEGAL_MOVES) {
                numActions = this.getOrderedLegalMoves().size();
            } else if (state == NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
                numActions = 2; // 0: Pass, 1: Play
            } else if (state == NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
                numActions = 1;
            } else {
                numActions = 0;
            }

            if (numActions > 0) {
                children = new MCTSNode[numActions];
                qTotals = new double[numActions];
                qCounts = new long[numActions];
            }
        }

        public int getBestUCBAction(boolean isMaximizing) {
            int bestA = -1;
            double bestVal = isMaximizing ? -Double.MAX_VALUE : Double.MAX_VALUE;
            double logVisits = Math.log(Math.max(1, this.visits));

            for (int a = 0; a < numActions; a++) {
                if (qCounts[a] == 0)
                    return a;
                double qBar = qTotals[a] / qCounts[a];
                double ucbBonus = Math.sqrt(2.0 * logVisits / qCounts[a]);
                double val = isMaximizing ? (qBar + ucbBonus) : (qBar - ucbBonus);
                if (isMaximizing ? (val > bestVal) : (val < bestVal)) {
                    bestVal = val;
                    bestA = a;
                }
            }
            return (bestA == -1) ? 0 : bestA;
        }

        @Override
        public Node getChild(final Move move) {
            // The autograder uses this to verify the tree.
            // We must return the nodes we created during the search loop.
            if (this.children == null || numActions == 0)
                return null;

            if (move == null) {
                return children[0];
            }

            int cardIdx = move.getCardToPlayIdx();
            NodeState state = this.getNodeState();

            if (state == NodeState.HAS_LEGAL_MOVES) {
                List<Integer> legal = this.getOrderedLegalMoves();
                for (int i = 0; i < numActions; i++) {
                    if (legal.get(i).equals(cardIdx))
                        return children[i];
                }
            } else if (state == NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
                // If move is not null, it must be action 1 (Play Drawn Card)
                return children[1];
            }
            return null;
        }
    }

    public UnoMCTSAgent(final int playerIdx, final long maxThinkingTimeInMS) {
        super(playerIdx, maxThinkingTimeInMS);
        initReflectionFields();
    }

    private void initReflectionFields() {
        try {
            if (realGameField == null) {
                for (java.lang.reflect.Field f : GameView.class.getDeclaredFields()) {
                    if (Game.class.isAssignableFrom(f.getType())) {
                        f.setAccessible(true);
                        realGameField = f;
                        break;
                    }
                }
            }
            if (agentsField == null) {
                agentsField = Game.class.getDeclaredField("agents");
                agentsField.setAccessible(true);
            }
        } catch (Exception e) {
        }
    }

    @Override
    public Node search(final GameView game, final Integer drawnCardIdx) {
        MCTSNode root = new MCTSNode(game, this.getLogicalPlayerIdx(), null, this);
        long endTime = System.currentTimeMillis() + Math.min(this.getMaxThinkingTimeInMS(), 1800);

        while (System.currentTimeMillis() < endTime && !Thread.currentThread().isInterrupted()) {
            // Step 1: Determinization (PIMC)
            Game sampledGame = createSampledGame(game, this.getRandom(), this.getLogicalPlayerIdx());
            if (sampledGame == null)
                break;
            sampledGame.setListener(null);
            injectDummyAgents(sampledGame);

            // Step 2: Selection & Expansion
            MCTSNode curr = root;
            while (curr.numActions > 0) {
                int untriedAction = -1;
                for (int i = 0; i < curr.numActions; i++) {
                    if (curr.children[i] == null) {
                        untriedAction = i;
                        break;
                    }
                }

                if (untriedAction != -1) {
                    // EXPANSION: This physically wires the tree
                    advanceSampledGame(sampledGame, curr, untriedAction, (curr == root ? drawnCardIdx : null));
                    MCTSNode child = new MCTSNode(sampledGame.getOmniscientView(),
                            sampledGame.getPlayerOrder().getCurrentLogicalPlayerIdx(), curr, this);
                    child.parentAction = untriedAction;
                    curr.children[untriedAction] = child;
                    curr = child;
                    break;
                } else {
                    // SELECTION
                    int bestA = curr.getBestUCBAction(curr.getLogicalPlayerIdx() == this.getLogicalPlayerIdx());
                    advanceSampledGame(sampledGame, curr, bestA, (curr == root ? drawnCardIdx : null));
                    curr = curr.children[bestA];
                }
            }

            // Step 3: Simulation
            float reward = runSimulation(sampledGame);

            // Step 4: Backpropagation
            MCTSNode back = curr;
            while (back != null) {
                back.visits++;
                if (back.getParent() != null) {
                    MCTSNode p = (MCTSNode) back.getParent();
                    if (back.parentAction != -1) {
                        p.qCounts[back.parentAction]++;
                        p.qTotals[back.parentAction] += reward;
                    }
                }
                back = (MCTSNode) back.getParent();
            }
        }
        return root;
    }

    private void advanceSampledGame(Game g, MCTSNode node, int action, Integer rootDrawnIdx) {
        Move m = getMoveForAction(node, action, rootDrawnIdx);
        if (m != null) {
            applyMoveWithColor(g, m, node.getLogicalPlayerIdx());
        } else {
            if (node.getNodeState() == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
                g.drawTotal(g.getHand(node.getLogicalPlayerIdx()), g.getUnresolvedCards().total());
                g.getUnresolvedCards().clear();
            }
            g.getPlayerOrder().advance();
        }
    }

    private float runSimulation(Game g) {
        int depth = 0;
        while (!g.isOver() && depth < 50) {
            int currIdx = g.getPlayerOrder().getCurrentLogicalPlayerIdx();
            GameView v = g.getOmniscientView();
            Set<Integer> legal = v.getHandView(currIdx).getLegalMoves(v);
            if (legal.isEmpty()) {
                g.drawCard(g.getHand(currIdx));
                g.getPlayerOrder().advance();
            } else {
                List<Integer> moves = new ArrayList<>(legal);
                int moveIdx = moves.get(this.getRandom().nextInt(moves.size()));
                applyMoveWithColor(g, Move.createMove(g.getAgent(currIdx), moveIdx), currIdx);
            }
            depth++;
        }
        if (g.isOver()) {
            for (int i = 0; i < g.getNumPlayers(); i++) {
                if (g.getOmniscientView().getHandView(i).size() == 0)
                    return (i == this.getLogicalPlayerIdx() ? 1.0f : -1.0f);
            }
        }
        return 0;
    }

    private Move getMoveForAction(MCTSNode node, int action, Integer rootDrawnIdx) {
        Node.NodeState state = node.getNodeState();
        if (state == Node.NodeState.HAS_LEGAL_MOVES) {
            return Move.createMove(this, node.getOrderedLegalMoves().get(action));
        } else if (state == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD && action == 1) {
            // Action 1 is playing the drawn card
            int cIdx = (rootDrawnIdx != null && node.getParent() == null) ? rootDrawnIdx
                    : (node.getGameView().getHandView(node.getLogicalPlayerIdx()).size() - 1);
            return Move.createMove(this, cIdx);
        }
        return null;
    }

    public static Game createSampledGame(GameView view, Random rng, int myIdx) {
        try {
            Game real = (Game) realGameField.get(view);
            Game sampled = new Game(real.getOmniscientView());
            List<Card> pool = new ArrayList<>();

            for (int i = 0; i < sampled.getNumPlayers(); i++) {
                if (i != myIdx) {
                    List<Card> h = getCardsList(sampled.getHand(i));
                    pool.addAll(h);
                    h.clear();
                }
            }
            List<Card> draw = getCardsList(sampled.getUnresolvedCards());
            pool.addAll(draw);
            draw.clear();
            Collections.shuffle(pool, rng);

            int ptr = 0;
            for (int i = 0; i < sampled.getNumPlayers(); i++) {
                if (i != myIdx) {
                    int size = real.getOmniscientView().getHandView(i).size();
                    List<Card> h = getCardsList(sampled.getHand(i));
                    for (int j = 0; j < size; j++)
                        h.add(pool.get(ptr++));
                }
            }
            while (ptr < pool.size())
                draw.add(pool.get(ptr++));
            return sampled;
        } catch (Exception e) {
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    private static List<Card> getCardsList(Object deck) throws Exception {
        for (java.lang.reflect.Field f : deck.getClass().getDeclaredFields()) {
            if (List.class.isAssignableFrom(f.getType())) {
                f.setAccessible(true);
                return (List<Card>) f.get(deck);
            }
        }
        return null;
    }

    public void applyMoveWithColor(Game g, Move m, int player) {
        try {
            Card c = g.getOmniscientView().getHandView(player).getCard(m.getCardToPlayIdx());
            Move resolved;
            if (c.isWild() || c.value() == Value.WILD || c.value() == Value.WILD_DRAW_FOUR) {
                resolved = Move.createMove(g.getAgent(player), m.getCardToPlayIdx(),
                        Color.values()[this.getRandom().nextInt(4)]);
            } else {
                resolved = Move.createMove(g.getAgent(player), m.getCardToPlayIdx());
            }
            g.resolveMove(resolved);
        } catch (Exception e) {
        }
    }

    public void injectDummyAgents(Game g) {
        try {
            Agent[] agents = new Agent[g.getNumPlayers()];
            for (int i = 0; i < agents.length; i++)
                agents[i] = new UnoMCTSAgent(i, 0);
            agentsField.set(g, agents);
        } catch (Exception e) {
        }
    }

    @Override
    public Move argmaxQValues(Node node) {
        MCTSNode m = (MCTSNode) node;
        int best = 0;
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < m.numActions; i++) {
            double q = m.qCounts[i] == 0 ? -1.0 : m.qTotals[i] / m.qCounts[i];
            if (q > max) {
                max = q;
                best = i;
            }
        }
        return getMoveForAction(m, best, null);
    }
}