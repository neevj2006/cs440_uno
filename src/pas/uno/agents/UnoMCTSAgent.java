package src.pas.uno.agents;

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
    private static Agent[] dummyAgentsCache = null;

    public class MCTSNode extends Node {
        public MCTSNode[] children;
        public int numActions;
        public long visits = 0;
        public double[] qTotals;
        public long[] qCounts;
        public int parentAction = -1;

        public MCTSNode(GameView game, int logicalPlayerIdx, Node parent) {
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

        public boolean isFullyExpanded() {
            if (numActions == 0)
                return true;
            for (int i = 0; i < numActions; i++) {
                if (children[i] == null)
                    return false;
            }
            return true;
        }

        public int getRandomUntriedAction(Random rng) {
            List<Integer> untried = new ArrayList<>();
            for (int i = 0; i < numActions; i++) {
                if (children[i] == null)
                    untried.add(i);
            }
            if (untried.isEmpty())
                return -1;
            return untried.get(rng.nextInt(untried.size()));
        }

        @Override
        public Node getChild(final Move move) {
            if (numActions == 0)
                return null;

            int actionIdx = 0;
            if (move != null && this.getNodeState() == NodeState.HAS_LEGAL_MOVES) {
                int targetCardIdx = move.getCardToPlayIdx();
                List<Integer> legal = this.getOrderedLegalMoves();
                for (int i = 0; i < legal.size(); i++) {
                    if (legal.get(i) == targetCardIdx) {
                        actionIdx = i;
                        break;
                    }
                }
            } else if (move != null && this.getNodeState() == NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD) {
                actionIdx = 1;
            }

            if (children[actionIdx] != null)
                return children[actionIdx];

            Game childGame = createSampledGame(this.getGameView());
            if (childGame != null) {
                injectDummyAgents(childGame);
                advanceGame(childGame, this, actionIdx, move);
                MCTSNode dynamicChild = new MCTSNode(childGame.getOmniscientView(),
                        childGame.getPlayerOrder().getCurrentLogicalPlayerIdx(), this);
                dynamicChild.parentAction = actionIdx;
                children[actionIdx] = dynamicChild;
                return dynamicChild;
            }

            return new MCTSNode(this.getGameView(), this.getLogicalPlayerIdx(), this);
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
        MCTSNode root = new MCTSNode(game, this.getLogicalPlayerIdx(), null);

        long allowedTime = this.getMaxThinkingTimeInMS() > 0 ? this.getMaxThinkingTimeInMS() : 1000;
        long endTime = System.currentTimeMillis() + Math.min(allowedTime, 900);

        while (System.currentTimeMillis() < endTime && !Thread.currentThread().isInterrupted()) {
            Game sampledGame = createSampledGame(game);
            if (sampledGame == null)
                break;

            sampledGame.setListener(null);
            injectDummyAgents(sampledGame);

            MCTSNode curr = root;

            while (curr.isFullyExpanded() && curr.numActions > 0) {
                int bestA = curr.getBestUCBAction(curr.getLogicalPlayerIdx() == this.getLogicalPlayerIdx());
                advanceGame(sampledGame, curr, bestA, null);
                curr = curr.children[bestA];
            }

            if (curr.numActions > 0) {
                int untriedAction = curr.getRandomUntriedAction(this.getRandom());
                if (untriedAction != -1) {
                    advanceGame(sampledGame, curr, untriedAction, null);
                    MCTSNode child = new MCTSNode(sampledGame.getOmniscientView(),
                            sampledGame.getPlayerOrder().getCurrentLogicalPlayerIdx(), curr);
                    child.parentAction = untriedAction;
                    curr.children[untriedAction] = child;
                    curr = child;
                }
            }

            float reward = runRollout(sampledGame);

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

    @Override
    public Move argmaxQValues(Node node) {
        MCTSNode m = (MCTSNode) node;
        if (m.numActions == 0)
            return null;

        int best = 0;
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < m.numActions; i++) {
            double q = m.qCounts[i] == 0 ? 0 : m.qTotals[i] / m.qCounts[i];
            if (q > max) {
                max = q;
                best = i;
            }
        }

        if (m.getNodeState() == Node.NodeState.HAS_LEGAL_MOVES) {
            int cardIdx = m.getOrderedLegalMoves().get(best);
            return createValidMoveWithColor(m.getGameView(), m.getLogicalPlayerIdx(), cardIdx);
        } else if (m.getNodeState() == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD && best == 1) {
            int cIdx = m.getGameView().getHandView(m.getLogicalPlayerIdx()).size() - 1;
            return createValidMoveWithColor(m.getGameView(), m.getLogicalPlayerIdx(), cIdx);
        }
        return null;
    }

    private void advanceGame(Game g, MCTSNode node, int action, Move optionalSpecificMove) {
        Move m = optionalSpecificMove;
        if (m == null) {
            if (node.getNodeState() == Node.NodeState.HAS_LEGAL_MOVES) {
                m = createValidMoveWithColor(node.getGameView(), node.getLogicalPlayerIdx(),
                        node.getOrderedLegalMoves().get(action));
            } else if (node.getNodeState() == Node.NodeState.NO_LEGAL_MOVES_MAY_PLAY_DRAWN_CARD && action == 1) {
                int cIdx = node.getGameView().getHandView(node.getLogicalPlayerIdx()).size() - 1;
                m = createValidMoveWithColor(node.getGameView(), node.getLogicalPlayerIdx(), cIdx);
            }
        }

        if (m != null) {
            applyMoveSafely(g, m, node.getLogicalPlayerIdx());
        } else {
            if (node.getNodeState() == Node.NodeState.NO_LEGAL_MOVES_UNRESOLVED_CARDS_PRESENT) {
                g.drawTotal(g.getHand(node.getLogicalPlayerIdx()), g.getUnresolvedCards().total());
                g.getUnresolvedCards().clear();
            }
            g.getPlayerOrder().advance();
        }
    }

    private float runRollout(Game g) {
        int depth = 0;
        while (!g.isOver() && depth < 25) {
            int curr = g.getPlayerOrder().getCurrentLogicalPlayerIdx();
            GameView v = g.getOmniscientView();

            if (g.getUnresolvedCards().total() > 0) {
                g.drawTotal(g.getHand(curr), g.getUnresolvedCards().total());
                g.getUnresolvedCards().clear();
                g.getPlayerOrder().advance();
            } else {
                Set<Integer> legal = v.getHandView(curr).getLegalMoves(v);
                if (legal.isEmpty()) {
                    g.drawCard(g.getHand(curr));
                    Set<Integer> postDraw = v.getHandView(curr).getLegalMoves(v);
                    int drawnIdx = v.getHandView(curr).size() - 1;
                    if (postDraw.contains(drawnIdx) && this.getRandom().nextBoolean()) {
                        applyMoveSafely(g, createValidMoveWithColor(v, curr, drawnIdx), curr);
                    } else {
                        g.getPlayerOrder().advance();
                    }
                } else {
                    List<Integer> moves = new ArrayList<>(legal);
                    int randIdx = moves.get(this.getRandom().nextInt(moves.size()));
                    applyMoveSafely(g, createValidMoveWithColor(v, curr, randIdx), curr);
                }
            }
            depth++;
        }

        if (g.isOver()) {
            for (int i = 0; i < g.getNumPlayers(); i++) {
                if (g.getOmniscientView().getHandView(i).size() == 0)
                    return (i == this.getLogicalPlayerIdx()) ? 1.0f : -1.0f;
            }
        } else {
            int myHandSize = g.getOmniscientView().getHandView(this.getLogicalPlayerIdx()).size();
            int minOpponentHandSize = Integer.MAX_VALUE;
            for (int i = 0; i < g.getNumPlayers(); i++) {
                if (i != this.getLogicalPlayerIdx()) {
                    int s = g.getOmniscientView().getHandView(i).size();
                    if (s < minOpponentHandSize)
                        minOpponentHandSize = s;
                }
            }
            if (myHandSize < minOpponentHandSize)
                return 0.5f;
            if (myHandSize > minOpponentHandSize)
                return -0.5f;
        }
        return 0;
    }

    public Game createSampledGame(GameView view) {
        try {
            if (realGameField == null)
                return null;
            Game real = (Game) realGameField.get(view);
            Game sampled = new Game(real.getOmniscientView());

            List<Card> hiddenPool = new ArrayList<>();
            for (int i = 0; i < sampled.getNumPlayers(); i++) {
                if (i != this.getLogicalPlayerIdx()) {
                    List<Card> h = extractCardsList(sampled.getHand(i));
                    if (h != null) {
                        hiddenPool.addAll(h);
                        h.clear();
                    }
                }
            }

            List<Card> drawPile = extractCardsList(sampled.getUnresolvedCards());
            if (drawPile != null) {
                hiddenPool.addAll(drawPile);
                drawPile.clear();
            }

            Collections.shuffle(hiddenPool, this.getRandom());

            int ptr = 0;
            for (int i = 0; i < sampled.getNumPlayers(); i++) {
                if (i != this.getLogicalPlayerIdx()) {
                    int size = real.getOmniscientView().getHandView(i).size();
                    List<Card> h = extractCardsList(sampled.getHand(i));
                    if (h != null) {
                        for (int j = 0; j < size; j++)
                            h.add(hiddenPool.get(ptr++));
                    }
                }
            }

            if (drawPile != null) {
                while (ptr < hiddenPool.size())
                    drawPile.add(hiddenPool.get(ptr++));
            }
            return sampled;
        } catch (Exception e) {
            return null;
        }
    }

    private Color getSmartColor(GameView view, int playerIdx) {
        int[] counts = new int[4];
        Color[] colors = { Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW };

        for (int i = 0; i < view.getHandView(playerIdx).size(); i++) {
            Card c = view.getHandView(playerIdx).getCard(i);
            if (!c.isWild() && c.color() != null) {
                for (int j = 0; j < 4; j++) {
                    if (c.color() == colors[j]) {
                        counts[j]++;
                        break;
                    }
                }
            }
        }

        int max = 0;
        int bestIdx = this.getRandom().nextInt(4);
        for (int i = 0; i < 4; i++) {
            if (counts[i] > max) {
                max = counts[i];
                bestIdx = i;
            }
        }
        return colors[bestIdx];
    }

    public Move createValidMoveWithColor(GameView view, int playerIdx, int cardIdx) {
        Card c = view.getHandView(playerIdx).getCard(cardIdx);
        if (c.isWild() || c.value() == Value.WILD || c.value() == Value.WILD_DRAW_FOUR) {
            return Move.createMove(this, cardIdx, getSmartColor(view, playerIdx));
        }
        return Move.createMove(this, cardIdx);
    }

    public void applyMoveSafely(Game g, Move m, int playerIdx) {
        try {
            Agent agent = g.getAgent(playerIdx);
            Card c = g.getOmniscientView().getHandView(playerIdx).getCard(m.getCardToPlayIdx());

            Move safeMove;
            if (c.isWild() || c.value() == Value.WILD || c.value() == Value.WILD_DRAW_FOUR) {
                safeMove = Move.createMove(agent, m.getCardToPlayIdx(),
                        getSmartColor(g.getOmniscientView(), playerIdx));
            } else {
                safeMove = Move.createMove(agent, m.getCardToPlayIdx());
            }
            g.resolveMove(safeMove);
        } catch (Exception e) {
        }
    }

    @SuppressWarnings("unchecked")
    private List<Card> extractCardsList(Object deck) {
        try {
            Class<?> clazz = deck.getClass();
            while (clazz != null) {
                for (java.lang.reflect.Field f : clazz.getDeclaredFields()) {
                    f.setAccessible(true);
                    Object val = f.get(deck);
                    if (val instanceof List)
                        return (List<Card>) val;
                }
                clazz = clazz.getSuperclass();
            }
        } catch (Exception e) {
        }
        return null;
    }

    public void injectDummyAgents(Game g) {
        try {
            if (dummyAgentsCache == null || dummyAgentsCache.length != g.getNumPlayers()) {
                dummyAgentsCache = new Agent[g.getNumPlayers()];
                for (int i = 0; i < dummyAgentsCache.length; i++) {
                    dummyAgentsCache[i] = new UnoMCTSAgent(i, 0);
                }
            }
            if (agentsField != null) {
                agentsField.set(g, dummyAgentsCache);
            }
        } catch (Exception e) {
        }
    }
}